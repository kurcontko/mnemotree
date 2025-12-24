from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime
import asyncio

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.memory import BaseMemory

from ...core.memory import MemoryCore
from ...core.query import MemoryQueryBuilder


class MemoryLangChainAdapter(BaseMemory):
    """Adapter to use Memory System as LangChain memory."""

    def __init__(
        self,
        memory_core: MemoryCore,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output"
    ):
        self.memory = memory_core
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load memory variables."""
        asyncio.run(self.aload_memory_variables(inputs))

    async def aload_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load memory variables."""
        # Get current input
        query = inputs.get(self.input_key, "")

        # Build query to get recent relevant memories
        memories = await self.memory.recall(
            query=query,
            limit=10,
        )

        # Format memories for chat context
        if memories:
            history = [memory.to_str_llm() for memory in memories]
        else:
            history = ""

        return {self.memory_key: history}

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> None:
        """Save context from conversation."""
        asyncio.run(self.asave_context(inputs, outputs))

    async def asave_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> None:
        """Save context from conversation."""
        # Get input/output
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")

        # Store as memory
        await self.memory.remember(
            content=f"User: {input_str}\nAssistant: {output_str}"
        )

    def clear(self) -> None:
        """Clear memory (optional)."""
        pass

    async def aclear(self) -> None:
        """Clear memory (optional)."""
        pass
