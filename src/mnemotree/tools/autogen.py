from __future__ import annotations

from typing import Any

from autogen import register_function

from ..core.memory import MemoryCore
from ..core.scoring import MemoryScoring


class AutogenMemoryTool:
    """Memory tools for Autogen agents with native async support."""

    def __init__(self, memory_core: MemoryCore):
        self.memory_core = memory_core
        self.scorer = MemoryScoring()

    @register_function(name="search_memories", description="Search through stored memories")
    async def search_memories(
        self, query: str, limit: int = 5, min_relevance: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Search through stored memories.

        Args:
            query (str): Search query.
            limit (int, optional): Maximum number of results. Defaults to 5.
            min_relevance (float, optional): Minimum relevance score (0-1). Defaults to 0.7.

        Returns:
            List[Dict[str, Any]]: List of matching memories.
        """
        # MemoryCore.recall does not accept `min_relevance`. We apply it here as a
        # post-filtering threshold using the same MemoryScoring logic.
        memories = await self.memory_core.recall(query, limit=None, scoring=False)
        query_embedding = await self.memory_core.get_embedding(query)

        scored = []
        for memory in memories:
            score = self.scorer.calculate_memory_score(
                memory,
                query_embedding=query_embedding,
            )
            if score >= min_relevance:
                scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [memory for _, memory in scored[:limit]]
        return [memory.model_dump() for memory in selected]

    @register_function(name="store_memory", description="Store a new memory")
    async def store_memory(self, content: str, importance: float, tags: list[str]) -> str:
        """
        Store a new memory.

        Args:
            content (str): Memory content.
            importance (float): Memory importance (0-1).
            tags (List[str]): Memory tags.

        Returns:
            str: Confirmation message.
        """
        memory = await self.memory_core.remember(
            content,
            importance=importance,
            tags=tags,
        )
        return memory.memory_id

    def get_tools(self) -> list[dict[str, Any]]:
        """Get all memory tools configured for Autogen."""
        return [
            {
                "name": "search_memories",
                "description": "Search through stored memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                        "min_relevance": {
                            "type": "number",
                            "description": "Minimum relevance score (0-1)",
                            "default": 0.7,
                        },
                    },
                    "required": ["query"],
                },
                "async_function": self.search_memories,
            },
            {
                "name": "store_memory",
                "description": "Store a new memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "importance": {"type": "number", "description": "Memory importance (0-1)"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Memory tags",
                        },
                    },
                    "required": ["content"],
                },
                "async_function": self.store_memory,
            },
        ]
