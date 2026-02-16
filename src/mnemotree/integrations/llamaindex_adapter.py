"""
LlamaIndex integration adapter for mnemotree.

Provides ChatMemoryBuffer and VectorMemory implementations
that use mnemotree as the backend storage.
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
    from llama_index.core.memory.types import BaseMemory

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Define stub types for type checking
    ChatMessage = Any
    MessageRole = Any
    ChatMemoryBuffer = Any
    BaseMemory = Any

from ..core import MemoryCore, RecallFilters
from ..core.models import MemoryType


class MnemotreeVectorMemory:
    """
    LlamaIndex memory adapter using mnemotree's vector storage.

    Provides semantic search over conversation history with
    automatic importance decay and memory consolidation.

    Example:
        ```python
        from llama_index.core.chat_engine import SimpleChatEngine
        from mnemotree import MemoryCore
        from mnemotree.store import ChromaMemoryStore
        from mnemotree.integrations import LlamaIndexMemoryAdapter

        store = ChromaMemoryStore()
        memory_core = MemoryCore(store=store)

        memory = MnemotreeVectorMemory(
            memory_core=memory_core,
            user_id="user-123"
        )

        # Use with LlamaIndex chat engine
        chat_engine = SimpleChatEngine.from_defaults(
            memory=memory,
            llm=llm
        )

        response = chat_engine.chat("What did I say about AI memory?")
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        user_id: str,
        max_messages: int = 10,
        memory_type: MemoryType = MemoryType.EPISODIC,
    ):
        """Initialize the vector memory.

        Args:
            memory_core: Mnemotree memory core instance
            user_id: Unique identifier for this user/session
            max_messages: Maximum number of messages to retrieve
            memory_type: Type of memory to store messages as
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for this adapter. Install it with: pip install llama-index"
            )

        self.memory_core = memory_core
        self.user_id = user_id
        self.max_messages = max_messages
        self.memory_type = memory_type

    async def aget(
        self,
        query: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Get relevant messages using semantic search.

        Args:
            query: Search query (if None, returns recent messages)
            **kwargs: Additional arguments

        Returns:
            List of ChatMessage objects
        """
        search_query = query or ""

        memories = await self.memory_core.recall(
            query=search_query,
            limit=self.max_messages,
            filters=RecallFilters(tags=[f"user:{self.user_id}"]),
        )

        messages = []
        for mem in memories:
            role_str = mem.metadata.get("role", "user")
            role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT

            messages.append(
                ChatMessage(
                    role=role,
                    content=mem.content,
                )
            )

        return messages

    def get(
        self,
        query: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Sync version of aget."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't use run_until_complete in async context
            return []

        return loop.run_until_complete(self.aget(query, **kwargs))

    async def aput(self, message: ChatMessage) -> None:
        """Store a message.

        Args:
            message: ChatMessage to store
        """
        role_str = "user" if message.role == MessageRole.USER else "assistant"

        await self.memory_core.remember(
            content=str(message.content),
            memory_type=self.memory_type,
            tags=[f"user:{self.user_id}", f"role:{role_str}"],
            metadata={"role": role_str},
        )

    def put(self, message: ChatMessage) -> None:
        """Sync version of aput."""
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.aput(message))

    async def areset(self) -> None:
        """Clear all messages for this user."""
        memories = await self.memory_core.recall(
            query="",
            limit=1000,
            filters=RecallFilters(tags=[f"user:{self.user_id}"]),
        )

        for mem in memories:
            await self.memory_core.forget(mem.memory_id)

    def reset(self) -> None:
        """Sync version of areset."""
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.areset())


class LlamaIndexMemoryAdapter:
    """
    Convenience wrapper for creating LlamaIndex-compatible memory.

    Example:
        ```python
        from mnemotree.integrations import LlamaIndexMemoryAdapter

        adapter = LlamaIndexMemoryAdapter(memory_core=memory_core)
        memory = adapter.create_vector_memory(user_id="user-123")

        # Use with any LlamaIndex component expecting memory
        ```
    """

    def __init__(self, memory_core: MemoryCore):
        """Initialize the adapter.

        Args:
            memory_core: Mnemotree memory core instance
        """
        self.memory_core = memory_core

    def create_vector_memory(
        self,
        user_id: str,
        max_messages: int = 10,
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> MnemotreeVectorMemory:
        """Create a vector memory instance.

        Args:
            user_id: Unique identifier for this user/session
            max_messages: Maximum number of messages to retrieve
            memory_type: Type of memory to store messages as

        Returns:
            MnemotreeVectorMemory instance
        """
        return MnemotreeVectorMemory(
            memory_core=self.memory_core,
            user_id=user_id,
            max_messages=max_messages,
            memory_type=memory_type,
        )
