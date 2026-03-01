"""Backward-compatibility re-exports.

All retrieval classes now live in ``mnemotree.core.retrieval``.
"""

from .retrieval import (
    FusionStrategy,
    HybridRetriever,
    RetrievalResult,
    RetrievalStage,
)

__all__ = [
    "RetrievalStage",
    "FusionStrategy",
    "RetrievalResult",
    "HybridRetriever",
]
