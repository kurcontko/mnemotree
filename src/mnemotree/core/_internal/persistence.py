from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...store.protocols import MemoryCRUDStore
from ..models import MemoryItem
from .indexing import IndexManager


@runtime_checkable
class Persistence(Protocol):
    async def save(self, memory: MemoryItem) -> None: ...
    async def delete(self, memory_id: str, cascade: bool = False) -> bool: ...


class DefaultPersistence:
    def __init__(self, store: MemoryCRUDStore, index_manager: IndexManager | None):
        self.store = store
        self.index_manager = index_manager

    async def save(self, memory: MemoryItem) -> None:
        await self.store.store_memory(memory)
        if self.index_manager:
            self.index_manager.add(memory)

    async def delete(self, memory_id: str, cascade: bool = False) -> bool:
        result = await self.store.delete_memory(memory_id, cascade=cascade)
        if result and self.index_manager:
            self.index_manager.remove(memory_id)
        return result
