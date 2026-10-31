"""One SQLite transaction owns sources, FTS5, vectors, and ingestion receipts."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from ..errors import ConfigurationError, IngestConflict
from .atom_store import AtomStore
from .atoms import Atom
from .lexical import create_fts, fts_table, index_text, match_expression
from .models import Chunk, ContextSpan, EmbeddingSpec, IngestReceipt, Source, digest, scope

# Re-exported: these lived here before schema.py and vectors.py.
from .schema import CHUNK_FTS as CHUNK_FTS
from .schema import SCHEMA_VERSION as SCHEMA_VERSION
from .schema import configure, connect, create_tables, stored_embedding_spec
from .vectors import vector_blob as vector_blob


class Window(NamedTuple):
    """The span of a source a retrieved chunk is served as (widened by neighbor chunks)."""

    source: Source
    start: int
    end: int


class IngestBlobs(NamedTuple):
    """Encoded vectors of one request: per chunk, and per atom."""

    chunks: list[bytes]
    atoms: list[bytes]


class SQLiteEvidenceStore:
    """Persistent local store. Exact cosine search filters the corpus before top-k.

    No ANN approximation or query-time reembedding is hidden in the baseline.
    Each instance owns a connection; WAL permits independent readers and writers.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_spec: EmbeddingSpec | None = None,
        extractor_config: dict[str, Any] | None = None,
    ):
        self._lock = threading.RLock()
        self.db = connect(path)
        try:
            create_tables(self.db)
            with self.transaction():
                self.embedding_spec = configure(self.db, embedding_spec)
            self.atoms = AtomStore(
                self.db, self.embedding_spec.dimensions if self.embedding_spec else None
            )
            with self.transaction():
                self.extractor_config = self.atoms.configure(extractor_config)
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self, *, read: bool = False) -> Iterator[None]:
        with self._lock:
            self.db.execute("BEGIN" if read else "BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def receipt(self, namespace: str, request_id: str, body_hash: str) -> IngestReceipt | None:
        scope(namespace)
        with self._lock:
            row = self.db.execute(
                "SELECT body_hash, receipt FROM evidence_requests WHERE namespace=? AND request_id=?",
                (namespace, request_id),
            ).fetchone()
            if row is None:
                return None
            if row["body_hash"] != body_hash:
                raise IngestConflict("request_id already belongs to a different payload")
            receipt = IngestReceipt.model_validate_json(row["receipt"])
            count = self.db.execute(
                "SELECT count(*) FROM evidence_sources WHERE namespace=? AND request_id=?",
                (namespace, request_id),
            ).fetchone()[0]
            if count != len(receipt.source_ids):
                raise IngestConflict(
                    "this request contains forgotten sources; replay cannot restore them"
                )
            return receipt.model_copy(update={"replayed": True})

    def ingest(
        self,
        namespace: str,
        request_id: str,
        session_id: str,
        body_hash: str,
        sources: Sequence[Source],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]] | None,
        atoms: Sequence[Atom] = (),
        atom_vectors: Sequence[Sequence[float]] | None = None,
    ) -> IngestReceipt:
        scope(namespace)
        blobs = self._ingest_blobs(
            namespace, request_id, session_id, sources, chunks, vectors, atoms, atom_vectors
        )
        with self.transaction():
            # Another process may have configured the file since this instance opened it.
            if stored_embedding_spec(self.db) != self.embedding_spec:
                raise ConfigurationError(
                    "database configuration changed; reopen this Memory instance"
                )
            if self.atoms.configure(self.extractor_config) != self.extractor_config:
                raise ConfigurationError(
                    "database configuration changed; reopen this Memory instance"
                )
            replay = self.receipt(namespace, request_id, body_hash)
            if replay:
                return replay
            try:
                self._insert(namespace, request_id, session_id, sources, chunks, blobs.chunks)
                self.atoms.insert(namespace, atoms, blobs.atoms)
            except sqlite3.IntegrityError as exc:
                raise IngestConflict(
                    "source IDs must identify distinct occurrences within a namespace"
                ) from exc
            receipt = IngestReceipt(
                namespace=namespace,
                request_id=request_id,
                session_id=session_id,
                source_ids=tuple(source.id for source in sources),
                chunk_count=len(chunks),
                atom_count=len(atoms),
            )
            self.db.execute(
                "INSERT INTO evidence_requests VALUES(?,?,?,?)",
                (namespace, request_id, body_hash, receipt.model_dump_json()),
            )
            return receipt

    def _ingest_blobs(
        self,
        namespace: str,
        request_id: str,
        session_id: str,
        sources: Sequence[Source],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]] | None,
        atoms: Sequence[Atom],
        atom_vectors: Sequence[Sequence[float]] | None,
    ) -> IngestBlobs:
        """Check a request is internally consistent; encode its vectors.

        Runs before the write transaction, so a malformed request never holds the lock.
        """
        if not sources or len({source.id for source in sources}) != len(sources):
            raise ValueError("a request needs sources with distinct IDs")
        if any(
            source.namespace != namespace
            or source.request_id != request_id
            or source.session_id != session_id
            for source in sources
        ):
            raise ValueError("source scope does not match ingestion request")
        source_map = {source.id: source for source in sources}
        for chunk in chunks:
            source = source_map.get(chunk.source_id)
            if source is None or not 0 <= chunk.start < chunk.end <= len(source.content):
                raise ValueError("invalid chunk source span")
            if source.content[chunk.start : chunk.end] != chunk.text:
                raise ValueError("chunk is not an exact source span")
        if {chunk.source_id for chunk in chunks} != set(source_map):
            raise ValueError("every source must have searchable chunks")
        blobs: list[bytes] = []
        if self.embedding_spec:
            if vectors is None or len(vectors) != len(chunks):
                raise ValueError("every chunk requires an embedding before acknowledging ingestion")
            blobs = [vector_blob(vector, self.embedding_spec.dimensions) for vector in vectors]
        elif vectors is not None:
            raise ValueError("vectors supplied without an embedding specification")
        atom_blobs: list[bytes] = []
        for atom in atoms:
            if self.extractor_config is None or atom.extractor_signature != digest(
                self.extractor_config
            ):
                raise ValueError("atom extractor signature does not match the database")
            if not atom.spans:
                raise ValueError("atoms require source evidence")
            for span in atom.spans:
                source = source_map.get(span.source_id)
                if source is None or not 0 <= span.start < span.end <= len(source.content):
                    raise ValueError("atom evidence must belong to this ingestion request")
        if self.embedding_spec and atoms:
            if atom_vectors is None or len(atom_vectors) != len(atoms):
                raise ValueError("every atom requires an embedding")
            atom_blobs = [
                vector_blob(vector, self.embedding_spec.dimensions) for vector in atom_vectors
            ]
        elif atom_vectors:
            raise ValueError("unexpected atom vectors")
        return IngestBlobs(blobs, atom_blobs)

    def _insert(
        self,
        namespace: str,
        request_id: str,
        session_id: str,
        sources: Sequence[Source],
        chunks: Sequence[Chunk],
        blobs: Sequence[bytes],
    ) -> None:
        """Write sources, chunks and their index rows; the caller holds the transaction."""
        next_seq = self.db.execute(
            "SELECT coalesce(max(session_seq) + 1, 0) FROM evidence_sources "
            "WHERE namespace=? AND session_id=?",
            (namespace, session_id),
        ).fetchone()[0]
        table = fts_table(CHUNK_FTS, namespace)
        if chunks:
            self.db.execute(create_fts(table))
        self.db.executemany(
            "INSERT INTO evidence_sources(namespace,id,request_id,session_id,position,"
            "session_seq,content,role,timestamp,ingested_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    namespace,
                    source.id,
                    request_id,
                    session_id,
                    source.position,
                    next_seq + offset,
                    source.content,
                    source.role,
                    source.timestamp.isoformat() if source.timestamp else None,
                    source.ingested_at.isoformat(),
                )
                for offset, source in enumerate(sources)
            ],
        )
        # Row by row: the FTS and vector rows are keyed by each chunk's new rowid.
        for index, chunk in enumerate(chunks):
            cursor = self.db.execute(
                "INSERT INTO evidence_chunks(namespace,id,source_id,chunk_index,start,end,text) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    namespace,
                    chunk.id,
                    chunk.source_id,
                    chunk.index,
                    chunk.start,
                    chunk.end,
                    chunk.text,
                ),
            )
            self.db.execute(
                f"INSERT INTO {table}(rowid,text) VALUES(?,?)",
                (cursor.lastrowid, index_text(chunk.text)),
            )
            if blobs:
                self.db.execute(
                    "INSERT INTO evidence_vec(chunk_rowid,namespace,embedding) VALUES(?,?,?)",
                    (cursor.lastrowid, namespace, blobs[index]),
                )

    def assistant_share(self, namespace: str) -> float:
        """Share of the namespace's message characters carried by assistant turns."""
        scope(namespace)
        with self._lock:
            row = self.db.execute(
                "SELECT COALESCE(SUM(CASE WHEN role='assistant' THEN LENGTH(content) END), 0), "
                "COALESCE(SUM(LENGTH(content)), 0) FROM evidence_sources WHERE namespace=?",
                (namespace,),
            ).fetchone()
        return int(row[0]) / int(row[1]) if row[1] else 0.0

    def chunk_density(self, namespace: str) -> float:
        """Chunks per stored message in the namespace (0.0 when empty)."""
        scope(namespace)
        with self._lock:
            row = self.db.execute(
                "SELECT (SELECT COUNT(*) FROM evidence_chunks WHERE namespace=?), "
                "(SELECT COUNT(*) FROM evidence_sources WHERE namespace=?)",
                (namespace, namespace),
            ).fetchone()
        chunks, sources = int(row[0]), int(row[1])
        return chunks / sources if sources else 0.0

    def get_source(self, namespace: str, source_id: str) -> Source | None:
        scope(namespace)
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM evidence_sources WHERE namespace=? AND id=?", (namespace, source_id)
            ).fetchone()
            return self._source(row) if row is not None else None

    def get_sources(self, namespace: str, source_ids: Sequence[str]) -> dict[str, Source]:
        """The stored sources among ``source_ids``, read in one snapshot; missing IDs are absent."""
        scope(namespace)
        ids = list(dict.fromkeys(source_ids))
        found: dict[str, Source] = {}
        with self.transaction(read=True):
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                rows = self.db.execute(
                    "SELECT * FROM evidence_sources WHERE namespace=? AND id IN ("
                    + ",".join("?" * len(batch))
                    + ")",
                    (namespace, *batch),
                ).fetchall()
                for row in rows:
                    found[row["id"]] = self._source(row)
        return found

    @staticmethod
    def _source(row: sqlite3.Row) -> Source:
        return Source.model_validate({key: row[key] for key in Source.model_fields})

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source_id=row["source_id"],
            index=row["chunk_index"],
            start=row["start"],
            end=row["end"],
            text=row["text"],
        )

    def source_chunks(self, namespace: str, source_ids: Sequence[str]) -> dict[str, list[Chunk]]:
        """Every chunk of the given source messages, by source, in chunk order."""
        scope(namespace)
        chunks: dict[str, list[Chunk]] = {source_id: [] for source_id in source_ids}
        ids = list(chunks)
        with self._lock:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                rows = self.db.execute(
                    "SELECT * FROM evidence_chunks WHERE namespace=? AND source_id IN ("
                    + ",".join("?" * len(batch))
                    + ") ORDER BY source_id, chunk_index",
                    (namespace, *batch),
                ).fetchall()
                for row in rows:
                    chunks[row["source_id"]].append(self._chunk(row))
        return chunks

    def lexical(
        self, namespace: str, query: str, limit: int, *, options: Sequence[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        scope(namespace)
        match = match_expression(query, options=options)
        if match is None:
            return []
        table = fts_table(CHUNK_FTS, namespace)
        with self._lock:
            if not self.db.execute(
                "SELECT 1 FROM evidence_chunks WHERE namespace=? LIMIT 1", (namespace,)
            ).fetchone():
                return []
            rows = self.db.execute(
                f"SELECT c.*, bm25({table}) AS score FROM {table} "
                f"JOIN evidence_chunks c ON c.rowid={table}.rowid "
                f"WHERE {table} MATCH ? AND c.namespace=? ORDER BY score,c.id LIMIT ?",
                (match, namespace, limit),
            ).fetchall()
        return [(self._chunk(row), -float(row["score"])) for row in rows]

    def dense(
        self, namespace: str, vector: Sequence[float], limit: int
    ) -> list[tuple[Chunk, float]]:
        scope(namespace)
        if self.embedding_spec is None:
            raise ConfigurationError("dense retrieval needs an indexed embedding model")
        blob = vector_blob(vector, self.embedding_spec.dimensions)
        with self._lock:
            rows = self.db.execute(
                "SELECT c.*, v.distance FROM (SELECT chunk_rowid, distance FROM evidence_vec "
                "WHERE embedding MATCH ? AND namespace=? AND k=?) v "
                "JOIN evidence_chunks c ON c.rowid=v.chunk_rowid ORDER BY v.distance,c.id",
                (blob, namespace, limit),
            ).fetchall()
        return [(self._chunk(row), 1.0 - float(row["distance"])) for row in rows]

    def window(
        self,
        namespace: str,
        chunk: Chunk,
        neighbors: int,
        *,
        source: Source | None = None,
    ) -> Window:
        """Materialize a chunk window, optionally reusing a source from this read snapshot."""
        scope(namespace)
        with self._lock:
            if source is None:
                source = self.get_source(namespace, chunk.source_id)
            if source is None:
                raise ValueError("source is not present in this namespace")
            if source.namespace != namespace or source.id != chunk.source_id:
                raise ValueError("source does not match this chunk's namespace and ID")
            if neighbors == 0:
                return Window(source, chunk.start, chunk.end)
            row = self.db.execute(
                "SELECT min(start),max(end) FROM evidence_chunks "
                "WHERE namespace=? AND source_id=? AND chunk_index BETWEEN ? AND ?",
                (namespace, chunk.source_id, chunk.index - neighbors, chunk.index + neighbors),
            ).fetchone()
            return Window(source, row[0], row[1])

    def forget(self, namespace: str, source_id: str) -> bool:
        scope(namespace)
        with self.transaction():
            # Read first: the cascade and triggers below remove these rows, and a trigger
            # cannot name a table per namespace.
            chunks = self.db.execute(
                "SELECT rowid FROM evidence_chunks WHERE namespace=? AND source_id=?",
                (namespace, source_id),
            ).fetchall()
            atoms = self.atoms.source_rows(namespace, source_id)
            cursor = self.db.execute(
                "DELETE FROM evidence_sources WHERE namespace=? AND id=?", (namespace, source_id)
            )
            if chunks:
                self.db.executemany(
                    f"DELETE FROM {fts_table(CHUNK_FTS, namespace)} WHERE rowid=?",
                    [tuple(row) for row in chunks],
                )
            self.atoms.unindex(namespace, atoms)
            return cursor.rowcount > 0

    def conversation_context(
        self,
        namespace: str,
        source: Source,
        *,
        neighbors: int,
        max_chars: int,
        source_cache: dict[str, Source] | None = None,
    ) -> tuple[ContextSpan, ...]:
        """Adjacent messages of the same session, in acknowledged ingestion order.

        AML splits one session into Add requests of at most 20 messages and sends
        them in source order, so context continues across request boundaries.
        Concurrent Adds to one session are ordered by commit, not by payload.
        A caller holding a read transaction can retain full sources in its snapshot cache.
        """
        scope(namespace)
        if source.namespace != namespace:
            raise ValueError("source is outside this namespace")
        if neighbors == 0:
            return ()
        with self._lock:
            query = (
                "SELECT * FROM evidence_sources WHERE namespace=? AND session_id=? "
                "AND session_seq{} ? ORDER BY session_seq {} LIMIT ?"
            )
            arguments = (namespace, source.session_id, source.session_seq, neighbors)
            before = self.db.execute(query.format("<", "DESC"), arguments).fetchall()
            after = self.db.execute(query.format(">", "ASC"), arguments).fetchall()
        rows = [(-index, row) for index, row in enumerate(before, 1)][::-1]
        rows += [(index, row) for index, row in enumerate(after, 1)]
        result = []
        for offset, row in rows:
            other = Source.model_validate({key: row[key] for key in Source.model_fields})
            if source_cache is not None:
                source_cache[other.id] = other
            start = max(0, len(other.content) - max_chars) if offset < 0 else 0
            end = min(len(other.content), start + max_chars)
            result.append(
                ContextSpan(
                    id=digest(["context", other.id, start, end]),
                    source_id=other.id,
                    request_id=other.request_id,
                    session_id=other.session_id,
                    position=other.position,
                    seq=other.session_seq,
                    offset=offset,
                    role=other.role,
                    timestamp=other.timestamp,
                    start=start,
                    end=end,
                    text=other.content[start:end],
                )
            )
        return tuple(result)

    def stats(self, namespace: str) -> dict[str, int]:
        scope(namespace)
        with self.transaction(read=True):
            return {
                table: self.db.execute(
                    f"SELECT count(*) FROM evidence_{table} WHERE namespace=?", (namespace,)
                ).fetchone()[0]
                for table in ("sources", "chunks", "requests")
            }


def now() -> datetime:
    return datetime.now(UTC)
