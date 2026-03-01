"""
Conflict detection and auto-invalidation for memory items.

When a new memory is stored, ConflictDetector finds semantically similar existing
memories. If the new memory supersedes an older one (higher timestamp + cosine
similarity above threshold), the older memory is invalidated via bi-temporal
valid_until field and a CONTRADICTS link is created.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import LinkType, MemoryItem
from .scoring import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConflictConfig:
    """Configuration for automatic conflict detection on remember()."""

    enable: bool = False
    similarity_threshold: float = 0.92
    candidate_limit: int = 10
    auto_invalidate: bool = True
    create_contradicts_link: bool = True


class ConflictDetector:
    """Detects and resolves conflicts when a new memory is stored.

    Phase 1 (fast, zero LLM): embedding cosine similarity + timestamp ordering.
    If a candidate has cosine similarity > threshold and the new memory is newer,
    the candidate is marked as invalidated (valid_until = now).
    """

    def __init__(self, config: ConflictConfig) -> None:
        self.config = config

    async def detect_and_resolve(
        self,
        new_memory: MemoryItem,
        store: object,
    ) -> list[str]:
        """Run conflict detection against the store and return invalidated memory IDs.

        Args:
            new_memory: The newly stored MemoryItem (must have embedding set).
            store: The MemoryCRUDStore instance.

        Returns:
            List of memory IDs that were invalidated.
        """
        from ..store.protocols import (
            SupportsInvalidation,
            SupportsKnowledgeGraph,
            SupportsVectorSearch,
        )

        if not self.config.enable:
            return []
        if new_memory.embedding is None:
            return []
        if not isinstance(store, SupportsVectorSearch):
            return []

        try:
            candidates = await store.get_similar_memories(
                query=new_memory.content,
                query_embedding=new_memory.embedding,
                top_k=self.config.candidate_limit + 1,
            )
        except Exception:
            logger.warning("ConflictDetector: get_similar_memories failed", exc_info=True)
            return []

        invalidated: list[str] = []

        for candidate in candidates:
            if candidate.memory_id == new_memory.memory_id:
                continue
            if candidate.valid_until is not None:
                continue
            if candidate.embedding is None:
                continue

            sim = cosine_similarity(new_memory.embedding, candidate.embedding)
            if sim < self.config.similarity_threshold:
                continue

            if new_memory.timestamp <= candidate.timestamp:
                continue

            logger.debug(
                "Conflict detected: %s supersedes %s (similarity=%.3f)",
                new_memory.memory_id,
                candidate.memory_id,
                sim,
            )

            if self.config.auto_invalidate and isinstance(store, SupportsInvalidation):
                try:
                    ok = await store.invalidate_memory(
                        candidate.memory_id,
                        reason=f"Superseded by {new_memory.memory_id} (similarity={sim:.3f})",
                    )
                    if ok:
                        invalidated.append(candidate.memory_id)
                except Exception:
                    logger.warning(
                        "ConflictDetector: failed to invalidate %s",
                        candidate.memory_id,
                        exc_info=True,
                    )

            if self.config.create_contradicts_link and isinstance(store, SupportsKnowledgeGraph):
                try:
                    await store.create_link(
                        new_memory.memory_id,
                        candidate.memory_id,
                        LinkType.CONTRADICTS,
                        strength=sim,
                        context="Auto-detected temporal conflict: new memory supersedes older",
                        metadata={"auto_conflict": True, "similarity": sim},
                    )
                except Exception:
                    logger.debug(
                        "ConflictDetector: failed to create CONTRADICTS link for %s->%s",
                        new_memory.memory_id,
                        candidate.memory_id,
                        exc_info=True,
                    )

        return invalidated
