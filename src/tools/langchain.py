import asyncio
from typing import List, Optional

from langchain.tools import Tool
from pydantic import BaseModel, Field

from ..core.memory import MemoryCore
from ..core.models import MemoryItem
from ..utils.memory_formatter import MemoryFormatter


class SearchMemoriesInput(BaseModel):
    """Input for searching memories."""

    query: str = Field(
        ...,
        description="The natural language query used to search through stored memories.",
    )


class StoreMemoryInput(BaseModel):
    """Input for storing a new memory."""

    data: str = Field(..., description="The data to be stored as a new memory.")


class LangchainMemoryTool:
    """Memory tools for LangChain agents."""

    def __init__(self, memory_core: MemoryCore):
        self.memory_core = memory_core
        self.memory_formatter = MemoryFormatter()

    def search_memories(self, query: str) -> str:
        return asyncio.run(self.asearch_memories(query))

    async def asearch_memories(self, query: str) -> str:
        memories: List[MemoryItem] = await self.memory_core.recall(query)
        return [memory.to_str_llm() for memory in memories]

    def store_memory(self, data: str) -> str:
        return asyncio.run(self.astore_memory(data))

    async def astore_memory(self, data: str) -> str:
        return await self.memory_core.remember(data)

    def get_tools(self) -> List[Tool]:
        """Get all memory tools configured for LangChain."""
        return [
            self.get_search_memories_tool(),
            self.get_store_memory_tool(),
        ]

    def get_search_memories_tool(self) -> Tool:
        """Get the search_memories tool."""
        return Tool.from_function(
            func=self.search_memories,
            coroutine=self.asearch_memories,
            name="search_memories",
            description="Search through stored memories using natural language",
            args_schema=SearchMemoriesInput,
        )
    
    def get_store_memory_tool(self) -> Tool:
        """Get the store_memory tool."""
        return Tool.from_function(
            func=self.store_memory,
            coroutine=self.astore_memory,
            name="store_memory",
            description="Store a new memory",
            args_schema=StoreMemoryInput,
        )