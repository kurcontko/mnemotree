"""Cross-encoder reranker implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.models import MemoryItem
from .base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker for high-quality semantic precision.

    Uses a cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2) to compute
    semantic similarity between query and candidate pairs. Cross-encoders
    process query-document pairs jointly, providing more accurate relevance
    scores than bi-encoders at the cost of speed.

    Requires: sentence-transformers (pip install sentence-transformers)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier for a cross-encoder model.
                        Default uses MS MARCO trained MiniLM which provides
                        good quality/speed tradeoff.
        """
        self.model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                ) from err

            self._model = CrossEncoder(self.model_name)

    async def rerank(
        self, query: str, candidates: list[MemoryItem], top_k: int | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """
        Rerank candidates using cross-encoder model.

        Args:
            query: The search query.
            candidates: List of candidate memories to rerank.
            top_k: Optional limit on results to return.

        Returns:
            List of (memory, score) tuples sorted by relevance score.
        """
        if not candidates:
            return []

        # Run model inference in thread to avoid blocking
        ranked = await asyncio.to_thread(self._rerank_sync, query, candidates)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def _rerank_sync(
        self, query: str, candidates: list[MemoryItem]
    ) -> list[tuple[MemoryItem, float]]:
        """Synchronous reranking logic."""
        self._load_model()
        assert self._model is not None

        # Prepare query-candidate pairs
        pairs = [(query, mem.content) for mem in candidates]

        # Compute cross-encoder scores
        scores = self._model.predict(pairs)

        # Pair memories with scores and sort by score descending
        ranked = list(zip(candidates, [float(s) for s in scores], strict=True))
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked
