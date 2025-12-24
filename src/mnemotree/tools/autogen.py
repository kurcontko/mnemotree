from __future__ import annotations

from typing import List, Dict, Any

from autogen import register_function

from ..core.memory import MemoryCore


class AutogenMemoryTool:
    """Memory tools for Autogen agents with native async support."""

    def __init__(self, memory_core: MemoryCore):
        self.memory_core = memory_core

    @register_function(name="search_memories", description="Search through stored memories")
    async def search_memories(self, query: str, limit: int = 5, min_relevance: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search through stored memories.

        Args:
            query (str): Search query.
            limit (int, optional): Maximum number of results. Defaults to 5.
            min_relevance (float, optional): Minimum relevance score (0-1). Defaults to 0.7.

        Returns:
            List[Dict[str, Any]]: List of matching memories.
        """
        return await self.memory_core.recall(query, limit=limit, min_relevance=min_relevance)

    @register_function(name="store_memory", description="Store a new memory")
    async def store_memory(self, content: str, importance: float, tags: List[str]) -> str:
        """
        Store a new memory.

        Args:
            content (str): Memory content.
            importance (float): Memory importance (0-1).
            tags (List[str]): Memory tags.

        Returns:
            str: Confirmation message.
        """
        return await self.memory_core.remember(content, importance=importance, tags=tags)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all memory tools configured for Autogen."""
        return [
            {
                "name": "search_memories",
                "description": "Search through stored memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Maximum number of results", "default": 5},
                        "min_relevance": {"type": "number", "description": "Minimum relevance score (0-1)", "default": 0.7}
                    },
                    "required": ["query"]
                },
                "async_function": self.search_memories
            },
            {
                "name": "store_memory",
                "description": "Store a new memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "importance": {"type": "number", "description": "Memory importance (0-1)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Memory tags"}
                    },
                    "required": ["content"]
                },
                "async_function": self.store_memory
            }
        ]
