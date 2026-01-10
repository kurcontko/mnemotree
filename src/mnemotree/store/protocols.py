from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.models import MemoryItem
from ..core.query import MemoryQuery


@runtime_checkable
class SupportsClose(Protocol):
    async def close(self) -> None: ...


@runtime_checkable
class MemoryCRUDStore(SupportsClose, Protocol):
    async def store_memory(self, memory: MemoryItem) -> None: ...

    async def get_memory(self, memory_id: str) -> MemoryItem | None: ...

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool: ...


@runtime_checkable
class SupportsMetadataUpdate(Protocol):
    async def update_memory_metadata(self, memory_id: str, metadata: dict[str, Any]) -> bool: ...


@runtime_checkable
class SupportsConnections(Protocol):
    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: list[str] | None = None,
        conflict_ids: list[str] | None = None,
        previous_id: str | None = None,
        next_id: str | None = None,
    ) -> None: ...


@runtime_checkable
class SupportsVectorSearch(Protocol):
    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsStructuredQuery(Protocol):
    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsEntityQuery(Protocol):
    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsMemoryListing(Protocol):
    async def list_memories(
        self,
        *,
        include_embeddings: bool = False,
    ) -> list[MemoryItem]: ...
