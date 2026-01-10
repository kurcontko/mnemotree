from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import MemoryItem
from ..scoring import MemoryScoring, cosine_similarity


def keyword_overlap_score(tags: list[str] | None, keywords: list[str] | None) -> float:
    if not tags or not keywords:
        return 0.0
    tag_tokens: set[str] = set()
    for tag in tags:
        lowered = tag.lower()
        tag_tokens.add(lowered)
        tag_tokens.update(lowered.replace("-", " ").split())
    keyword_tokens = {kw.lower() for kw in keywords if kw}
    if not keyword_tokens:
        return 0.0
    return len(tag_tokens & keyword_tokens) / len(keyword_tokens)


@runtime_checkable
class Ranker(Protocol):
    def rank(
        self,
        memories: list[MemoryItem],
        query_embedding: list[float] | None,
        extra_signals: dict[str, Any] | None,
    ) -> list[MemoryItem]: ...


class MemoryScorer:
    """Adapter for MemoryScoring to the Ranker protocol."""

    def __init__(self, scoring: MemoryScoring):
        self.scoring = scoring

    def rank(
        self,
        memories: list[MemoryItem],
        query_embedding: list[float] | None,
        extra_signals: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return self.scoring.filter_memories_by_score(
            memories,
            query_embedding=query_embedding,
        )


class SignalRanker:
    """Ranks memories based on composite signals (RRF, Entity, Keywords)."""

    def __init__(
        self, rrf_weight: float = 0.15, entity_boost: float = 0.05, keyword_boost: float = 0.10
    ):
        self.rrf_weight = rrf_weight
        self.entity_boost = entity_boost
        self.keyword_boost = keyword_boost

    def rank(
        self,
        memories: list[MemoryItem],
        query_embedding: list[float] | None,
        extra_signals: dict[str, Any] | None,
    ) -> list[MemoryItem]:
        if not memories:
            return memories

        extra_signals = extra_signals or {}
        rrf_scores = extra_signals.get("rrf_scores", {})
        query_keywords = extra_signals.get("query_keywords", [])
        entity_memory_ids = extra_signals.get("entity_memory_ids", set())

        max_rrf = max(rrf_scores.values()) if rrf_scores else 0.0

        def score(memory: MemoryItem) -> float:
            similarity = max(0.0, cosine_similarity(memory.embedding, query_embedding))
            keyword_score = keyword_overlap_score(memory.tags, query_keywords)
            entity_hit = 1.0 if memory.memory_id in entity_memory_ids else 0.0
            rrf_norm = (rrf_scores.get(memory.memory_id, 0.0) / max_rrf) if max_rrf else 0.0
            return (
                similarity
                + (self.rrf_weight * rrf_norm)
                + (self.entity_boost * entity_hit)
                + (self.keyword_boost * keyword_score)
            )

        return sorted(
            memories,
            key=lambda memory: (
                score(memory),
                cosine_similarity(memory.embedding, query_embedding),
            ),
            reverse=True,
        )
