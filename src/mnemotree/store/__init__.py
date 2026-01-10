from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import BaseMemoryStore
from .protocols import (
    MemoryCRUDStore,
    SupportsConnections,
    SupportsEntityQuery,
    SupportsMemoryListing,
    SupportsMetadataUpdate,
    SupportsStructuredQuery,
    SupportsVectorSearch,
)

if TYPE_CHECKING:
    from .chromadb_store import ChromaMemoryStore
    from .milvus_store import MilvusMemoryStore
    from .neo4j_store import Neo4jMemoryStore
    from .sqlite_vec_store import SQLiteVecMemoryStore


__all__ = [
    "BaseMemoryStore",
    "MemoryCRUDStore",
    "SupportsConnections",
    "SupportsEntityQuery",
    "SupportsMemoryListing",
    "SupportsMetadataUpdate",
    "SupportsStructuredQuery",
    "SupportsVectorSearch",
    "ChromaMemoryStore",
    "MilvusMemoryStore",
    "Neo4jMemoryStore",
    "SQLiteVecMemoryStore",
]


_LAZY_EXPORTS: dict[str, tuple[str, str, str]] = {
    "ChromaMemoryStore": (".chromadb_store", "ChromaMemoryStore", "mnemotree[chroma]"),
    "MilvusMemoryStore": (".milvus_store", "MilvusMemoryStore", "mnemotree[milvus]"),
    "Neo4jMemoryStore": (".neo4j_store", "Neo4jMemoryStore", "mnemotree[neo4j]"),
    "SQLiteVecMemoryStore": (
        ".sqlite_vec_store",
        "SQLiteVecMemoryStore",
        "mnemotree[sqlite_vec]",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, symbol_name, extra = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(module_name, package=__name__)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{name} requires optional dependencies. Install with `{extra}`."
            ) from exc
        return getattr(module, symbol_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
