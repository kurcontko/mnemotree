"""Legacy core exports, resolved lazily so stores can import model contracts first."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..rerankers import CrossEncoderReranker, NoOpReranker
    from .builder import MemoryCoreBuilder
    from .hybrid_retrieval import FusionStrategy, HybridRetriever, RetrievalStage
    from .memory import (
        IngestionConfig,
        MemoryCore,
        MemoryMode,
        ModeDefaultsConfig,
        NerConfig,
        RecallFilters,
        RecallOptions,
        RememberOptions,
        RetrievalConfig,
        RetrievalMode,
        ScoringConfig,
    )
    from .models import LinkType, MemoryItem, MemoryLink, MemoryType
    from .query import MemoryQuery, MemoryQueryBuilder
    from .retriever_factory import RetrieverFactory

__all__ = [
    "MemoryCore",
    "MemoryCoreBuilder",
    "MemoryMode",
    "MemoryItem",
    "MemoryLink",
    "MemoryQuery",
    "MemoryQueryBuilder",
    "MemoryType",
    "LinkType",
    "RememberOptions",
    "RecallFilters",
    "RecallOptions",
    "IngestionConfig",
    "ModeDefaultsConfig",
    "NerConfig",
    "RetrievalConfig",
    "RetrievalMode",
    "ScoringConfig",
    "HybridRetriever",
    "CrossEncoderReranker",
    "NoOpReranker",
    "FusionStrategy",
    "RetrievalStage",
    "RetrieverFactory",
]

_EXPORTS = {
    "MemoryCoreBuilder": ".builder",
    **dict.fromkeys(
        (
            "MemoryCore",
            "MemoryMode",
            "RememberOptions",
            "RecallFilters",
            "RecallOptions",
            "IngestionConfig",
            "ModeDefaultsConfig",
            "NerConfig",
            "RetrievalConfig",
            "RetrievalMode",
            "ScoringConfig",
        ),
        ".memory",
    ),
    **dict.fromkeys(("MemoryItem", "MemoryLink", "MemoryType", "LinkType"), ".models"),
    **dict.fromkeys(("MemoryQuery", "MemoryQueryBuilder"), ".query"),
    **dict.fromkeys(("HybridRetriever", "FusionStrategy", "RetrievalStage"), ".hybrid_retrieval"),
    **dict.fromkeys(("CrossEncoderReranker", "NoOpReranker"), "..rerankers"),
    "RetrieverFactory": ".retriever_factory",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
