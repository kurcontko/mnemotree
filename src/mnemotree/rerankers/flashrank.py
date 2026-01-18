"""FlashRank reranker implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.models import MemoryItem
from .base import BaseReranker


class FlashRankReranker(BaseReranker):
    """
    FlashRank reranker for fast, lightweight reranking.

    FlashRank is optimized for speed while maintaining reasonable quality.
    It's a good choice when latency is important or for initial prototyping.

    Requires: flashrank (pip install mnemotree[rerank_flashrank])
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2") -> None:
        """
        Initialize FlashRank reranker.

        Args:
            model_name: FlashRank model name. Default uses TinyBERT which
                        provides fastest inference with good quality.
        """
        self.model_name = model_name
        self._ranker: Any | None = None
        self._request_cls: Any | None = None

    def _load(self) -> None:
        """Lazy load the FlashRank model on first use."""
        if self._ranker is not None:
            return
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError as exc:
            raise ImportError(
                "flashrank is required for FlashRankReranker. "
                "Install with: pip install mnemotree[rerank_flashrank]"
            ) from exc

        self._request_cls = RerankRequest
        try:
            self._ranker = Ranker(model_name=self.model_name)
        except TypeError:
            # Fallback for older flashrank versions
            self._ranker = Ranker(self.model_name)

    @staticmethod
    def _coerce_id(value: object) -> int | None:
        """Convert result ID to integer index."""
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    async def rerank(
        self, query: str, candidates: list[MemoryItem], top_k: int | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """
        Rerank candidates using FlashRank.

        Args:
            query: The search query.
            candidates: List of candidate memories to rerank.
            top_k: Optional limit on results to return.

        Returns:
            List of (memory, score) tuples sorted by relevance score.
        """
        if not candidates:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, candidates, top_k)

    def _rerank_sync(
        self, query: str, candidates: list[MemoryItem], top_k: int | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """Synchronous reranking logic."""
        self._load()
        assert self._ranker is not None
        assert self._request_cls is not None

        # Prepare passages with indices for tracking
        passages = [{"id": idx, "text": mem.content} for idx, mem in enumerate(candidates)]
        request = self._request_cls(query=query, passages=passages)
        results = self._ranker.rerank(request)

        # Build ranked list with scores
        ranked: list[tuple[MemoryItem, float]] = []
        seen: set[int] = set()

        for item in results:
            # Handle both dict and object result formats
            if isinstance(item, dict):
                result_id = item.get("id")
                score = item.get("score", 0.0)
            else:
                result_id = getattr(item, "id", None)
                score = getattr(item, "score", 0.0)

            idx = self._coerce_id(result_id)
            if idx is None or idx < 0 or idx >= len(candidates) or idx in seen:
                continue

            ranked.append((candidates[idx], float(score)))
            seen.add(idx)

        # Append any unseen candidates with score 0
        for idx, mem in enumerate(candidates):
            if idx not in seen:
                ranked.append((mem, 0.0))

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked
