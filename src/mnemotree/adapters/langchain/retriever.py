from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ...core.memory import MemoryCore


class MemoryRetriever(BaseRetriever):
    """Adapter to use Memory System as a LangChain retriever."""

    def __init__(self, memory_core: MemoryCore, search_kwargs: dict[str, Any] = None):
        super().__init__()
        self.memory = memory_core
        self.search_kwargs = search_kwargs or {"limit": 5, "min_importance": 0.6}

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Get relevant documents based on query."""
        k = kwargs.get("k", 10)

        # Get similar memories
        memories = await self.memory.recall(query=query, limit=k)

        # Convert to documents
        return [memory.to_langchain_document() for memory in memories]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Sync version of document retrieval."""
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager, **kwargs))
