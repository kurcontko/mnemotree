from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

MemorySource = Literal["conversation", "tool", "document", "system", "user"]


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: MemorySource = "conversation"

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryAnnotations(BaseModel):
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    entities: Dict[str, str] = Field(default_factory=dict)
    entity_mentions: Dict[str, List[str]] = Field(default_factory=dict)

    emotions: List[str] = Field(default_factory=list)
    linked_concepts: List[str] = Field(default_factory=list)

    importance: float = 0.5
    confidence: float = 1.0


class MemoryEdge(BaseModel):
    src_id: str
    dst_id: str
    kind: Literal[
        "related",
        "conflicts",
        "previous",
        "next",
        "mentions",
        "about",
    ]
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEnvelope(BaseModel):
    record: MemoryRecord
    ann: MemoryAnnotations = Field(default_factory=MemoryAnnotations)

    embedding: Optional[List[float]] = None
    edges: List[MemoryEdge] = Field(default_factory=list)


class MemoryFilter(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    tags_any: List[str] = Field(default_factory=list)
    min_importance: Optional[float] = None


class MemoryQuery(BaseModel):
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    filters: MemoryFilter = Field(default_factory=MemoryFilter)
    limit: int = 10

    hops: int = 0
    edge_kinds: Optional[List[str]] = None


class MemoryHit(BaseModel):
    memory: MemoryEnvelope
    score: float
    reasons: Dict[str, Any] = Field(default_factory=dict)

