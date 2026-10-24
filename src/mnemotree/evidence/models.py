"""Small immutable contracts for source ingestion and evidence retrieval."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mnemotree.evidence.temporal import annotate


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(Record):
    content: str = Field(min_length=1)
    role: str = Field(default="user", min_length=1, max_length=100)
    id: str | None = Field(default=None, min_length=1, max_length=256)
    timestamp: datetime | None = None

    @field_validator("content", "role", "id")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("timestamp")
    @classmethod
    def aware_time(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None:
                raise ValueError("timestamp must include a timezone")
            return value.astimezone(UTC)
        return value


class EmbeddingSpec(Record):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimensions: int = Field(gt=0, le=16384)

    @property
    def signature(self) -> str:
        return digest(self.model_dump())


class IngestReceipt(Record):
    # Receipts are stored and re-read on replay; a field added by a newer
    # version must not make an older one reject its own database.
    model_config = ConfigDict(frozen=True, extra="ignore")

    namespace: str
    request_id: str
    session_id: str
    source_ids: tuple[str, ...]
    chunk_count: int
    atom_count: int = 0
    replayed: bool = False


class Source(Record):
    namespace: str
    id: str
    request_id: str
    session_id: str
    position: int
    content: str
    role: str
    timestamp: datetime | None
    ingested_at: datetime
    # Assigned by the store at ingestion; ordering within the session.
    session_seq: int | None = None


class Chunk(Record):
    id: str
    source_id: str
    index: int
    start: int
    end: int
    text: str


class ChannelScore(Record):
    rank: int
    score: float


class Candidate(Record):
    chunk: Chunk
    fusion_score: float
    channels: dict[str, ChannelScore]
    rerank_score: float | None = None


class ContextSpan(Record):
    id: str
    source_id: str
    request_id: str
    session_id: str
    position: int
    seq: int | None
    # Messages before (<0) or after (>0) the anchor in session ingestion order.
    offset: int
    role: str
    timestamp: datetime | None
    start: int
    end: int
    text: str


class Evidence(Record):
    id: str
    source_id: str
    session_id: str
    role: str
    timestamp: datetime | None
    start: int
    end: int
    text: str
    score: float
    channels: dict[str, ChannelScore]
    rerank_score: float | None = None
    request_id: str | None = None
    position: int | None = None
    seq: int | None = None
    neighbors: tuple[ContextSpan, ...] = ()
    # Lower-ranked anchors from the same stretch of conversation, returned as
    # one item so a top_k cap counts passages rather than individual turns.
    merged: tuple[Evidence, ...] = ()

    def members(self) -> tuple[Evidence, ...]:
        return (self.model_copy(update={"merged": ()}), *self.merged)

    def source_ids(self) -> list[str]:
        return list(
            dict.fromkeys(
                source_id
                for member in self.members()
                for source_id in (member.source_id, *(n.source_id for n in member.neighbors))
            )
        )

    def render(self) -> str:
        """Plain transcript lines; dates and roles live in content for the reader.

        Identifiers and offsets stay in the trace: they cost answer tokens and
        give the answer model nothing to reason with. Overlapping spans of one
        message are joined, and "..." marks skipped messages within a block.
        """
        by_source: dict[str, list[Evidence | ContextSpan]] = {}
        for member in (self, *self.merged):
            for span in (member, *member.neighbors):
                by_source.setdefault(span.source_id, []).append(span)
        lines: list[tuple[tuple[int, int], int | None, str]] = []
        for spans in by_source.values():
            spans.sort(key=lambda span: (span.start, -span.end))
            joined: list[tuple[Evidence | ContextSpan, int, int, str]] = []
            for span in spans:
                if joined and span.start <= joined[-1][2]:
                    head, start, end, text = joined[-1]
                    if span.end > end:
                        joined[-1] = (head, start, span.end, text + span.text[end - span.start :])
                else:
                    joined.append((span, span.start, span.end, span.text))
            for head, start, _, text in joined:
                order = (head.seq if head.seq is not None else 0, start)
                lines.append((order, head.seq, render_line(head.timestamp, head.role, text)))
        lines.sort(key=lambda line: line[0])
        rendered: list[str] = []
        for index, (_, seq, line) in enumerate(lines):
            previous = lines[index - 1][1] if index else None
            if seq is not None and previous is not None and seq - previous > 1:
                rendered.append("...")
            rendered.append(line)
        return "\n".join(rendered)


def render_line(timestamp: datetime | None, role: str, text: str) -> str:
    # Equal instants in different zones print differently, so the offset is part of the key.
    offset = timestamp.utcoffset() if timestamp is not None else None
    return _render_line(timestamp, offset, role, text)


# Packing re-renders a block for every merge and neighbor it tries; the date notes are
# the costly part of a line, so each line is rendered once.
@lru_cache(maxsize=1 << 14)
def _render_line(
    timestamp: datetime | None, _offset: timedelta | None, role: str, text: str
) -> str:
    if timestamp is None:
        return f"{role}: {text}"
    # Weekday helps resolve "last Friday"; computed directly to stay locale-free.
    day = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[timestamp.weekday()]
    return f"[{timestamp:%Y-%m-%d} {day} {timestamp:%H:%M} UTC] {role}: {annotate(text, timestamp)}"


class SearchOptions(Record):
    representation: Literal["raw", "atoms", "combined"] = "raw"
    atom_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    candidate_limit: int = Field(default=100, ge=1, le=2000)
    token_budget: int = Field(default=8000, ge=0, le=117760)
    neighbor_chunks: int = Field(default=0, ge=0, le=2)
    neighbor_messages: int = Field(default=0, ge=0, le=2)
    neighbor_chars: int = Field(default=1200, ge=1, le=12000)
    # Return anchors within reach of each other in one session as a single item.
    merge_blocks: bool = False
    # Order of the returned items: by relevance, or chronologically. Selection is
    # by relevance either way; "time" only changes how the reader sees them.
    order: Literal["rank", "time"] = "rank"
    rrf_k: int = Field(default=60, ge=1)
    lexical_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    dense_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    # Keep at most this many fused candidates per source message: long assistant
    # turns otherwise occupy several candidate slots and several returned items.
    chunks_per_source: int | None = Field(default=None, ge=1)
    # Anchors of this role are selected before the others (each group keeps its
    # fusion order). Labelled evidence in long assistant/user histories is mostly
    # the user's short turns while most tokens are the assistant's.
    role_first: str | None = None
    # Use a deeper candidate pool when the user's history averages more than
    # deep_history_density chunks per message (long assistant turns crowd the pool).
    deep_candidate_limit: int | None = Field(default=None, ge=1, le=2000)
    deep_history_density: float = Field(default=2.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def enough_candidates(self) -> SearchOptions:
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be at least limit")
        if self.deep_candidate_limit is not None and self.deep_candidate_limit < self.limit:
            raise ValueError("deep_candidate_limit must be at least limit")
        return self


class SearchTrace(Record):
    mode: str
    candidates: tuple[Candidate, ...]
    atom_hits: tuple[dict, ...] = ()
    context_candidates: tuple[ContextSpan, ...] = ()
    omitted_context_ids: tuple[str, ...] = ()
    channel_counts: dict[str, int]
    packed_tokens: int
    tokenizer: str
    omitted_ids: tuple[str, ...]
    timings_ms: dict[str, float]
    # Pool size each channel actually used (after the history-density gate), and the
    # measured chunks per message when that gate was evaluated.
    candidate_limit: int = 0
    chunk_density: float | None = None


class SearchResult(Record):
    evidence: tuple[Evidence, ...]
    context: str
    trace: SearchTrace


def digest(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def scope(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("namespace must be a nonblank string of at most 256 characters")
    return value
