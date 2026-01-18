"""Base classes for reranker implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.models import MemoryItem


class BaseReranker(ABC):
    """Abstract base class for reranking strategies.

    Rerankers take a query and a list of candidate memories, then return
    the candidates reordered by relevance with associated scores.

    All reranker implementations should inherit from this class and
    implement the `rerank` method.
    """

    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[MemoryItem], top_k: int | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """
        Rerank candidates based on query relevance.

        Args:
            query: The search query to rank candidates against.
            candidates: List of candidate memories to rerank.
            top_k: Optional limit on number of results to return.
                   If None, returns all candidates reranked.

        Returns:
            List of (memory, score) tuples sorted by score descending.
            Scores should be in [0, 1] range where higher is more relevant.
        """


class NoOpReranker(BaseReranker):
    """Pass-through reranker that preserves original ordering.

    Useful for testing, baselines, or when reranking overhead is not desired.
    Assigns a score of 1.0 to all candidates.
    """

    async def rerank(
        self, query: str, candidates: list[MemoryItem], top_k: int | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """Return candidates with placeholder scores, preserving original order."""
        results = [(mem, 1.0) for mem in candidates]
        if top_k is not None:
            results = results[:top_k]
        return results
