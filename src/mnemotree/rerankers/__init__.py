"""Reranker backends for retrieval result reordering.

This module provides pluggable reranker implementations that can be used
to reorder retrieval results based on query relevance.

Available backends:
    - FlashRankReranker: Fast, lightweight reranker using FlashRank
    - CrossEncoderReranker: High-quality reranker using cross-encoder models
    - NoOpReranker: Pass-through reranker for testing/baseline
"""

from __future__ import annotations

from .base import BaseReranker, NoOpReranker
from .cross_encoder import CrossEncoderReranker
from .flashrank import FlashRankReranker


def create_reranker(backend: str, **kwargs: object) -> BaseReranker:
    """
    Create a reranker backend by name.

    Args:
        backend: Name of the reranker backend to create.
        **kwargs: Backend-specific configuration options.

    Returns:
        Configured reranker instance.

    Raises:
        ValueError: If the backend name is not recognized.

    Examples:
        >>> reranker = create_reranker("flashrank")
        >>> reranker = create_reranker("cross-encoder", model_name="ms-marco-MiniLM-L-6-v2")
        >>> reranker = create_reranker("noop")
    """
    key = backend.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"flashrank", "flash_rank"}:
        return FlashRankReranker(**kwargs)  # type: ignore[arg-type]
    if key in {"cross_encoder", "crossencoder", "ce"}:
        return CrossEncoderReranker(**kwargs)  # type: ignore[arg-type]
    if key in {"noop", "no_op", "none", "passthrough"}:
        return NoOpReranker()
    raise ValueError(f"Unknown reranker backend: {backend!r}")


__all__ = [
    "BaseReranker",
    "CrossEncoderReranker",
    "FlashRankReranker",
    "NoOpReranker",
    "create_reranker",
]
