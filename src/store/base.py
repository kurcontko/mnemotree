from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.models import MemoryItem, MemoryType


class BaseMemoryStorage(ABC):
    """Abstract base class for memory storage implementations."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the storage connection with provided configuration."""
        pass
    
    @abstractmethod
    def _setup_database(self):
        """Set up necessary database structures (tables, collections, indices)."""
        pass

    @abstractmethod
    def store_memory(self, memory: MemoryItem) -> None:
        """
        Store a memory item with all its relationships.
        
        Args:
            memory: MemoryItem to store
        """
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory item by its ID.
        
        Args:
            memory_id: Unique identifier of the memory
            
        Returns:
            MemoryItem if found, None otherwise
        """
        pass

    @abstractmethod
    def get_similar_memories(self, query_embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        """
        Retrieve similar memories using vector similarity search.
        
        Args:
            query_embedding: Vector to compare against
            top_k: Number of similar memories to retrieve
            
        Returns:
            List of similar MemoryItems
        """
        pass

    @abstractmethod
    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """
        Retrieve all memories of a specific type.
        
        Args:
            memory_type: Type of memories to retrieve
            
        Returns:
            List of matching MemoryItems
        """
        pass

    @abstractmethod
    def update_memory_importance(self, memory_id: str, current_time: datetime) -> None:
        """
        Update memory importance based on decay and access.
        
        Args:
            memory_id: ID of memory to update
            current_time: Current timestamp for decay calculation
        """
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory item and all its relationships.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if memory was deleted, False otherwise
        """
        pass

    @abstractmethod
    def close(self):
        """Close the storage connection."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # Optional methods that implementations might want to override
    def batch_store_memories(self, memories: List[MemoryItem]) -> None:
        """
        Store multiple memories efficiently.
        Default implementation calls store_memory for each item.
        
        Args:
            memories: List of MemoryItems to store
        """
        for memory in memories:
            self.store_memory(memory)

    def get_memories_by_query(self, query: Dict[str, Any]) -> List[MemoryItem]:
        """
        Retrieve memories matching a custom query.
        
        Args:
            query: Implementation-specific query parameters
            
        Returns:
            List of matching MemoryItems
        """
        raise NotImplementedError("Custom queries not implemented for this storage")

    def get_connected_memories(self, memory_id: str, relationship_type: str = None) -> List[MemoryItem]:
        """
        Retrieve memories connected to the given memory.
        
        Args:
            memory_id: ID of the source memory
            relationship_type: Type of relationship to traverse
            
        Returns:
            List of connected MemoryItems
        """
        raise NotImplementedError("Connected memory retrieval not implemented")
        
    def create_backup(self, backup_path: str) -> bool:
        """
        Create a backup of the memory store.
        
        Args:
            backup_path: Path to store the backup
            
        Returns:
            True if backup was successful, False otherwise
        """
        raise NotImplementedError("Backup not implemented for this storage")

    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_path: Path to the backup
            
        Returns:
            True if restore was successful, False otherwise
        """
        raise NotImplementedError("Restore not implemented for this storage")