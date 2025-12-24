from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Awaitable
from datetime import datetime

from ..core.models import MemoryItem, MemoryType
from ..core.query import MemoryQuery


class BaseMemoryStore(ABC):
    """
    Abstract base class for memory store implementations.
    Designed to support both vector and graph databases with vector capabilities.
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
        pass

    # @abstractmethod
    # async def batch_store_memories(self, memories: List[MemoryItem]) -> None:
    #     """
    #     Store multiple memory items efficiently.

    #     Args:
    #         memories (List[MemoryItem]): List of memory items to store.
    #     """
    #     pass

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory item by its ID.

        Args:
            memory_id (str): The unique identifier of the memory.

        Returns:
            Optional[MemoryItem]: The retrieved memory item or None if not found.
        """
        pass

    @abstractmethod
    async def delete_memory(
        self,
        memory_id: str,
        *,
        cascade: bool = False
    ) -> bool:
        """
        Delete a memory item by its ID.

        Args:
            memory_id (str): The unique identifier of the memory.
            cascade (bool, optional): If True, delete related connections. Defaults to False.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        pass

    # --------------------
    # Update Operations
    # --------------------

    # @abstractmethod
    # async def update_memory_metadata(
    #     self,
    #     memory_id: str,
    #     metadata: Dict[str, Any]
    # ) -> bool:
    #     """
    #     Update metadata of a memory item.

    #     Args:
    #         memory_id (str): The unique identifier of the memory.
    #         metadata (Dict[str, Any]): The metadata to update.

    #     Returns:
    #         bool: True if update was successful, False otherwise.
    #     """
    #     pass

    @abstractmethod
    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: Optional[List[str]] = None,
        conflict_ids: Optional[List[str]] = None,
        previous_id: Optional[str] = None,
        next_id: Optional[str] = None
    ) -> None:
        """
        Update connections of a memory item.

        Args:
            memory_id (str): The unique identifier of the memory.
            related_ids (Optional[List[str]], optional): IDs of related memories. Defaults to None.
            conflict_ids (Optional[List[str]], optional): IDs of conflicting memories. Defaults to None.
            previous_id (Optional[str], optional): ID of the previous memory in a sequence. Defaults to None.
            next_id (Optional[str], optional): ID of the next memory in a sequence. Defaults to None.
        """
        pass

    # --------------------
    # Retrieval Operations
    # --------------------

    @abstractmethod
    async def get_similar_memories(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        Retrieve memories similar to a given embedding.

        Args:
            query_embedding (List[float]): The embedding vector for similarity comparison.
            top_k (int, optional): Number of top similar memories to retrieve. Defaults to 5.
            filters (Optional[Dict[str, Any]], optional): Additional filters to apply. Defaults to None.

        Returns:
            List[MemoryItem]: List of similar memory items.
        """
        pass
    
    @abstractmethod
    async def query_by_entities(self, entities: Dict[str, str]) -> List[MemoryItem]:
        pass

    # @abstractmethod
    # async def search_memories(
    #     self,
    #     query_embedding: List[float],
    #     filters: Optional[Dict[str, Any]] = None,
    #     limit: int = 10
    # ) -> List[Tuple[MemoryItem, float]]:
    #     """
    #     Search memories based on vector similarity with scoring.

    #     Args:
    #         query_embedding (List[float]): The embedding vector for similarity comparison.
    #         filters (Optional[Dict[str, Any]], optional): Additional filters to apply. Defaults to None.
    #         limit (int, optional): Maximum number of results to return. Defaults to 10.

    #     Returns:
    #         List[Tuple[MemoryItem, float]]: List of tuples containing memory items and their similarity scores.
    #     """
    #     pass

    @abstractmethod
    async def query_memories(
        self,
        query: MemoryQuery,
    ) -> List[MemoryItem]:
        """
        Query memories using complex filter conditions and vector similarity.

        Args:
            query (MemoryQuery): The memory query specifications.

        Returns:
            List[MemoryItem]: List of memory items matching the query.
        """
        pass

    # @abstractmethod
    # async def get_memories_by_type(
    #     self,
    #     memory_type: MemoryType,
    #     limit: Optional[int] = None
    # ) -> List[MemoryItem]:
    #     """
    #     Retrieve memories of a specific type.

    #     Args:
    #         memory_type (MemoryType): The type of memories to retrieve.
    #         limit (Optional[int], optional): Maximum number of memories to retrieve. Defaults to None.

    #     Returns:
    #         List[MemoryItem]: List of memory items of the specified type.
    #     """
    #     pass

    # --------------------
    # Connection Management
    # --------------------

    # @abstractmethod
    # async def get_connected_memories(
    #     self,
    #     memory_id: str,
    #     *,
    #     connection_type: Optional[str] = None,
    #     limit: Optional[int] = None
    # ) -> List[MemoryItem]:
    #     """
    #     Retrieve memories connected to a given memory.

    #     Args:
    #         memory_id (str): The unique identifier of the memory.
    #         connection_type (Optional[str], optional): Type of connections to filter. Defaults to None.
    #         limit (Optional[int], optional): Maximum number of connected memories to retrieve. Defaults to None.

    #     Returns:
    #         List[MemoryItem]: List of connected memory items.
    #     """
    #     pass

    # --------------------
    # Utility Methods
    # --------------------

    # async def create_backup(self, backup_path: str) -> bool:
    #     """
    #     Create a backup of the storage.

    #     Args:
    #         backup_path (str): The file path where the backup will be stored.

    #     Returns:
    #         bool: True if backup was successful, False otherwise.
    #     """
    #     raise NotImplementedError("Backup not implemented")

    # async def restore_backup(self, backup_path: str) -> bool:
    #     """
    #     Restore storage from a backup.

    #     Args:
    #         backup_path (str): The file path of the backup to restore.

    #     Returns:
    #         bool: True if restoration was successful, False otherwise.
    #     """
    #     raise NotImplementedError("Restore not implemented")

    # async def get_statistics(self) -> Dict[str, Any]:
    #     """
    #     Retrieve statistics about the storage.

    #     Returns:
    #         Dict[str, Any]: A dictionary containing storage statistics.
    #     """
    #     raise NotImplementedError("Statistics not implemented")

    # async def optimize_storage(self) -> bool:
    #     """
    #     Optimize the storage for performance (e.g., rebuild indices).

    #     Returns:
    #         bool: True if optimization was successful, False otherwise.
    #     """
    #     raise NotImplementedError("Optimization not implemented")

    # --------------------
    # Context Management
    # --------------------

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
        pass

    # --------------------
    # Additional Helper Methods
    # --------------------

    # async def retrieve_and_update(
    #     self,
    #     memory_id: str,
    #     update_func: Callable[[MemoryItem], MemoryItem]
    # ) -> Optional[MemoryItem]:
    #     """
    #     Retrieve a memory, apply an update function, and save the updated memory.

    #     Args:
    #         memory_id (str): The unique identifier of the memory.
    #         update_func (Callable[[MemoryItem], MemoryItem]): A function that takes a MemoryItem and returns an updated MemoryItem.

    #     Returns:
    #         Optional[MemoryItem]: The updated memory item or None if retrieval failed.
    #     """
    #     memory = await self.get_memory(memory_id)
    #     if memory is None:
    #         return None
    #     updated_memory = update_func(memory)
    #     await self.store_memory(updated_memory)
    #     return updated_memory

    # async def exists(self, memory_id: str) -> bool:
    #     """
    #     Check if a memory exists in storage.

    #     Args:
    #         memory_id (str): The unique identifier of the memory.

    #     Returns:
    #         bool: True if the memory exists, False otherwise.
    #     """
    #     memory = await self.get_memory(memory_id)
    #     return memory is not None
