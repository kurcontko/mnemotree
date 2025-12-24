from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .models_v2 import MemoryEdge, MemoryEnvelope, MemoryFilter


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> List[float]: ...


@runtime_checkable
class Enricher(Protocol):
    async def enrich(self, env: MemoryEnvelope) -> MemoryEnvelope: ...


@runtime_checkable
class DocStore(Protocol):
    async def upsert(self, env: MemoryEnvelope) -> None: ...
    async def get(self, memory_id: str) -> Optional[MemoryEnvelope]: ...
    async def delete(self, memory_id: str) -> bool: ...


@runtime_checkable
class VectorIndex(Protocol):
    async def upsert(
        self, memory_id: str, embedding: List[float], payload: Dict[str, Any]
    ) -> None: ...

    async def query(
        self, embedding: List[float], filters: MemoryFilter, limit: int
    ) -> List[Tuple[str, float]]: ...


@runtime_checkable
class GraphStore(Protocol):
    async def upsert_edges(self, edges: List[MemoryEdge]) -> None: ...

    async def neighbors(
        self, memory_ids: List[str], hops: int, edge_kinds: Optional[List[str]]
    ) -> List[str]: ...

