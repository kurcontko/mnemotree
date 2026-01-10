from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.models import MemoryItem
from ..core.query import MemoryQuery


class BaseMemoryStore(ABC):
    """
    Abstract base class for memory store implementations.
    Designed to support both vector and graph databases with vector capabilities.

    Note:
        This ABC is retained for backward compatibility. New code should prefer
        the capability-based Protocol interfaces in `mnemotree.store.protocols`
        (e.g. `MemoryCRUDStore`, `SupportsVectorSearch`, `SupportsStructuredQuery`).

    Error handling convention:
        - Return False for expected "not found" or no-op updates.
        - Raise for storage/transport errors after logging with logger.exception.
    """

    # --------------------
    # CRUD Operations
    # --------------------

    @abstractmethod
    async def store_memory(self, memory: MemoryItem) -> None:
        """
        Store a single memory item.

        Args:
            memory (MemoryItem): The memory item to store.
        """

    @abstractmethod
    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """
        Retrieve a memory item by its ID.

        Args:
            memory_id (str): The unique identifier of the memory.

        Returns:
            Optional[MemoryItem]: The retrieved memory item or None if not found.
        """

    @abstractmethod
    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool:
        """
        Delete a memory item by its ID.

        Args:
            memory_id (str): The unique identifier of the memory.
            cascade (bool, optional): If True, delete related connections. Defaults to False.

        Returns:
            bool: True if deletion was successful, False when the memory does not exist.
        """

    # --------------------
    # Update Operations
    # --------------------

    async def update_memory_metadata(self, memory_id: str, metadata: dict[str, Any]) -> bool:
        """Update metadata fields for a memory item.

        Default implementation using get→modify→store pattern.
        Stores with optimized update paths should override this.

        Args:
            memory_id: The unique identifier of the memory.
            metadata: Dict of field names to new values.

        Returns:
            bool: True if update was successful, False when the memory does not exist
            or the update is a no-op.
        """
        if not metadata:
            return False
        memory = await self.get_memory(memory_id)
        if not memory:
            return False
        updated = False
        for key, value in metadata.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
                updated = True
        if not updated:
            return False
        await self.store_memory(memory)
        return True

    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: list[str] | None = None,
        conflict_ids: list[str] | None = None,
        previous_id: str | None = None,
        next_id: str | None = None,
    ) -> None:
        """Optionally update connections of a memory item.

        Graph-capable stores should override this.
        """
        raise NotImplementedError("Connections are not supported by this store")

    # --------------------
    # Retrieval Operations
    # --------------------

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """
        Retrieve memories similar to a given embedding.

        Args:
            query_embedding (List[float]): The embedding vector for similarity comparison.
            top_k (int, optional): Number of top similar memories to retrieve. Defaults to 5.
            filters (Optional[Dict[str, Any]], optional): Additional filters to apply. Defaults to None.

        Returns:
            List[MemoryItem]: List of similar memory items.
        """
        raise NotImplementedError("Vector similarity search is not supported by this store")

    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]:
        raise NotImplementedError("Entity queries are not supported by this store")

    async def query_memories(
        self,
        query: MemoryQuery,
    ) -> list[MemoryItem]:
        """
        Query memories using complex filter conditions and vector similarity.

        Args:
            query (MemoryQuery): The memory query specifications.

        Returns:
            List[MemoryItem]: List of memory items matching the query.
        """
        raise NotImplementedError("Structured queries are not supported by this store")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def close(self):
        """
        Close the storage connection gracefully.
        """
