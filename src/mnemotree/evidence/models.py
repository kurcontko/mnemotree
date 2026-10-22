"""Small immutable contracts for source ingestion and evidence retrieval."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
            return value.astimezone(timezone.utc)
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
    role: str
    timestamp: datetime | None
    start: int
    end: int
    text: str

    def render_value(self) -> dict:
        return {
            "source_id": self.source_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "position": self.position,
            "role": self.role,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "span": [self.start, self.end],
            "text": self.text,
        }


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
    neighbors: tuple[ContextSpan, ...] = ()

    def render(self) -> str:
        """Dates and attribution live in content, not transport-only metadata."""
        value: dict[str, object] = {
            "source_id": self.source_id,
            "session_id": self.session_id,
            "role": self.role,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "span": [self.start, self.end],
            "text": self.text,
        }
        if self.request_id is not None:
            value.update(request_id=self.request_id, position=self.position)
        if self.neighbors:
            value["neighbors"] = [neighbor.render_value() for neighbor in self.neighbors]
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )


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
    rrf_k: int = Field(default=60, ge=1)
    lexical_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    dense_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def enough_candidates(self) -> SearchOptions:
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be at least limit")
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
