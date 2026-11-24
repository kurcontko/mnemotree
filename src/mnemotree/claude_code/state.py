"""Short SQLite transactions for prompt receipts, delivered spans and the turn queue."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from ..core.models import Evidence, Message, SourceSpan, digest
from ..errors import IngestConflict
from ..memory import Memory
from .transcript import Submission, accepted_prompts, fingerprint, transcript_size

MESSAGES = TypeAdapter(list[Message])
STALE_SECONDS = 30 * 24 * 3600
# Claude Code writes the transcript asynchronously, so a Stop can arrive before the
# turn's records are on disk. Seconds to wait between the attempts to read them.
TRANSCRIPT_WAITS = (0.2, 0.5)

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions(
    session TEXT, key TEXT, fingerprint TEXT, offset INTEGER, submitted TEXT,
    PRIMARY KEY(session, key)
);
CREATE TABLE IF NOT EXISTS turns(
    seq INTEGER PRIMARY KEY, session TEXT, request_id TEXT UNIQUE, messages BLOB, finished REAL
);
CREATE INDEX IF NOT EXISTS pending_turns ON turns(seq) WHERE messages IS NOT NULL;
CREATE TABLE IF NOT EXISTS offered(
    session TEXT, key TEXT, source TEXT, start INTEGER, end INTEGER, used REAL,
    PRIMARY KEY(session, key, source, start, end)
);
CREATE TABLE IF NOT EXISTS seen(
    session TEXT, source TEXT, start INTEGER, end INTEGER, used INTEGER,
    PRIMARY KEY(session, source, start, end)
);
CREATE TABLE IF NOT EXISTS context_resets(session TEXT PRIMARY KEY, at TEXT);
"""


class HookState:
    """One outbox per absolute evidence database and namespace."""

    def __init__(self, directory: Path, db: sqlite3.Connection) -> None:
        self.directory = directory
        self.db = db

    @classmethod
    @contextmanager
    def open(cls, state: Path, database: Path, namespace: str) -> Iterator[HookState]:
        directory = state / digest([str(database), namespace])
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.mkdir(exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(directory / "hooks.sqlite", timeout=5)) as db:
            db.execute("PRAGMA secure_delete=ON")
            db.executescript(SCHEMA)
            yield HookState(directory, db)

    def submit(self, session: str, key: str, prompt: str, transcript: Path) -> None:
        # Another hook may still reject this submission. Only the digest is staged.
        with self.db:
            if not key:  # Clients without prompt IDs can correlate only their latest submit.
                self.db.execute("DELETE FROM submissions WHERE session=? AND key=''", (session,))
            self.db.execute(
                "INSERT OR IGNORE INTO submissions VALUES(?,?,?,?,?)",
                (
                    session,
                    key,
                    fingerprint(prompt),
                    transcript_size(transcript),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def discard(self, session: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM submissions WHERE session=?", (session,))

    def confirm(self, session: str, key: str, transcript: Path, reply: str | None) -> None:
        rows = self.db.execute(
            "SELECT key,fingerprint,offset,submitted FROM submissions WHERE session=? "
            "AND rowid <= (SELECT rowid FROM submissions WHERE session=? AND key=?) ORDER BY rowid",
            (session, session, key),
        ).fetchall()
        pending = [Submission(*row) for row in rows]
        if not pending:
            return
        result = self._read_transcript(transcript, session, pending)
        if result is None:
            return
        uuid, messages = result
        if reply is not None and reply.strip():
            messages.append(Message(role="assistant", content=reply, timestamp=datetime.now(UTC)))
        # Claim the exact receipts we read. A delayed or repeated Stop must not
        # consume a newer submission, including one from a client without IDs.
        with self.db:
            claimed = self.db.execute(
                "DELETE FROM submissions WHERE session=? AND key=? AND submitted=?",
                (session, key, pending[-1].submitted),
            )
            if not claimed.rowcount:
                return
            self.db.executemany(
                "DELETE FROM submissions WHERE session=? AND key=? AND submitted=?",
                ((session, item.key, item.submitted) for item in pending[:-1]),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO turns(session,request_id,messages) VALUES(?,?,?)",
                (session, digest(["claude-code", session, uuid]), MESSAGES.dump_json(messages)),
            )

    @staticmethod
    def _read_transcript(
        transcript: Path, session: str, pending: Sequence[Submission]
    ) -> tuple[str, list[Message]] | None:
        """The confirmed prompts, waiting briefly for a transcript that trails the Stop."""
        for wait in (*TRANSCRIPT_WAITS, None):
            try:
                result = accepted_prompts(transcript, session, pending)
            except OSError:
                if wait is None:
                    raise
                result = None
            if result is not None or wait is None:
                return result
            time.sleep(wait)
        return None

    def reset_context(self, session: str) -> None:
        """Claude's context was compacted or cleared: what it held is no longer there.

        Passages delivered before can be delivered again, and from now on only the
        session's later turns count as already in front of Claude.
        """
        with self.db:
            self.db.execute("DELETE FROM seen WHERE session=?", (session,))
            self.db.execute("DELETE FROM offered WHERE session=?", (session,))
            self.db.execute(
                "INSERT OR REPLACE INTO context_resets VALUES(?,?)",
                (session, datetime.now(UTC).isoformat()),
            )

    def context_start(self, session: str) -> datetime | None:
        """When the session's context last started over, or None if it never did."""
        row = self.db.execute(
            "SELECT at FROM context_resets WHERE session=?", (session,)
        ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def has_turns(self) -> bool:
        return (
            self.db.execute("SELECT 1 FROM turns WHERE messages IS NOT NULL LIMIT 1").fetchone()
            is not None
        )

    def flush(self, opener: Callable[[], Memory], report_error: Callable[[str], None]) -> None:
        import fcntl

        # Workers serialize only with other workers. No SQLite transaction spans
        # the provider call, so submissions can continue while the model loads.
        with (self.directory / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.has_turns():
                with opener() as memory:
                    self._drain(memory, report_error)
            self.prune()

    def _drain(self, memory: Memory, report_error: Callable[[str], None]) -> None:
        # Snapshot only IDs. An outage can leave many large turns in the queue.
        turns = self.db.execute(
            "SELECT seq,session,request_id FROM turns WHERE messages IS NOT NULL ORDER BY seq"
        ).fetchall()
        failed_sessions = set()
        for seq, session, request_id in turns:
            if session in failed_sessions:
                continue
            (payload,) = self.db.execute(
                "SELECT messages FROM turns WHERE seq=?", (seq,)
            ).fetchone()
            try:
                asyncio.run(
                    memory.remember_many(
                        MESSAGES.validate_json(payload),
                        session=session,
                        request_id=request_id,
                    )
                )
            except IngestConflict as exc:
                # A committed turn subsequently forgotten cannot be replayed.
                report_error(f"Retiring turn {request_id}: {exc}")
            except Exception as exc:  # noqa: BLE001
                report_error(f"{type(exc).__name__}: {exc}")
                failed_sessions.add(session)  # Keep later turns of this session in order.
                continue
            with self.db:
                self.db.execute(
                    "UPDATE turns SET messages=NULL, finished=? WHERE seq=?", (time.time(), seq)
                )

    def prune(self) -> None:
        cutoff = time.time() - STALE_SECONDS
        stale = datetime.fromtimestamp(cutoff, UTC).isoformat()
        with self.db:
            self.db.execute("DELETE FROM submissions WHERE submitted<?", (stale,))
            self.db.execute("DELETE FROM context_resets WHERE at<?", (stale,))
            self.db.execute("DELETE FROM seen WHERE used<?", (cutoff,))
            self.db.execute("DELETE FROM offered WHERE used<?", (cutoff,))
            self.db.execute("DELETE FROM turns WHERE finished<?", (cutoff,))

    def seen(self, session: str) -> list[SourceSpan]:
        rows = self.db.execute(
            "SELECT source,start,end FROM seen WHERE session=? ORDER BY source,start,end",
            (session,),
        )
        # Adjacent windows can together cover a third window.
        spans: list[SourceSpan] = []
        for source, start, end in rows:
            if spans and spans[-1].source_id == source and spans[-1].end >= start:
                prior = spans.pop()
                start, end = prior.start, max(prior.end, end)
            spans.append(SourceSpan(source_id=source, start=start, end=end))
        return spans

    def offer(self, session: str, key: str, evidence: Sequence[Evidence]) -> None:
        used = time.time()
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO offered VALUES(?,?,?,?,?,?)",
                (
                    (session, key, span.source_id, span.start, span.end, used)
                    for block in evidence
                    for member in block.members()
                    for span in member.spans()
                ),
            )

    def accept_offer(self, session: str, key: str) -> None:
        # A submit rejected by a sibling hook never delivered its passages to Claude.
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO seen SELECT session,source,start,end,used FROM offered "
                "WHERE session=? AND key=?",
                (session, key),
            )
            self.db.execute("DELETE FROM offered WHERE session=? AND key=?", (session, key))

    def clear_offer(self, session: str, key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM offered WHERE session=? AND key=?", (session, key))
