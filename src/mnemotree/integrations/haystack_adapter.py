"""
Haystack integration adapter for mnemotree.

Provides conversational memory for Haystack RAG agents with
chat history, semantic search, and FSRS-4.5 decay.

Reference: https://haystack.deepset.ai/
"""

from __future__ import annotations

from typing import Any

from ..core import MemoryCore, RecallFilters
from ..core.models import MemoryType


class MnemotreeHaystackMemory:
    """
    Haystack-compatible conversational memory using mnemotree.

    Provides:
    - ChatMessageStore-compatible interface
    - Semantic search over conversation history
    - FSRS-4.5 decay for message importance
    - RAG context enhancement

    Example:
        ```python
        from haystack import Pipeline
        from haystack.components.builders import ChatPromptBuilder
        from mnemotree import MemoryCore
        from mnemotree.store import ChromaMemoryStore
        from mnemotree.integrations import MnemotreeHaystackMemory

        store = ChromaMemoryStore()
        memory_core = MemoryCore(store=store)

        # Create memory for conversation
        chat_memory = MnemotreeHaystackMemory(
            memory_core=memory_core,
            session_id="conversation_123"
        )

        # Use in Haystack pipeline
        pipeline = Pipeline()
        pipeline.add_component("memory", chat_memory)
        pipeline.add_component("prompt", ChatPromptBuilder())

        # Run with automatic memory
        result = pipeline.run({
            "memory": {"query": "What did we discuss?"},
            "prompt": {"template": chat_template}
        })
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        session_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_messages: int = 50,
    ):
        """Initialize Haystack memory adapter.

        Args:
            memory_core: Mnemotree memory core instance
            session_id: Unique conversation session ID
            memory_type: Type of memory to store messages as
            max_messages: Maximum messages to retrieve
        """
        self.memory_core = memory_core
        self.session_id = session_id
        self.memory_type = memory_type
        self.max_messages = max_messages

    async def add_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional metadata
        """
        meta = metadata or {}
        meta["role"] = role

        await self.memory_core.remember(
            content=content,
            memory_type=self.memory_type,
            tags=[f"session:{self.session_id}", f"role:{role}"],
            metadata=meta,
        )

    async def get_messages(self, limit: int | None = None) -> list[dict]:
        """Retrieve conversation messages.

        Args:
            limit: Max messages (uses max_messages if None)

        Returns:
            List of message dicts with role and content
        """
        limit = limit or self.max_messages

        memories = await self.memory_core.recall(
            query="", limit=limit, filters=RecallFilters(tags=[f"session:{self.session_id}"])
        )

        return [
            {"role": (mem.metadata or {}).get("role", "user"), "content": mem.content}
            for mem in memories
        ]

    async def search_relevant(self, query: str, limit: int = 5) -> list[dict]:
        """Search for relevant past messages.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of relevant messages with metadata
        """
        memories = await self.memory_core.recall(
            query=query, limit=limit, filters=RecallFilters(tags=[f"session:{self.session_id}"])
        )

        return [
            {
                "role": (mem.metadata or {}).get("role", "user"),
                "content": mem.content,
                "timestamp": mem.timestamp.isoformat()
                if hasattr(mem.timestamp, "isoformat")
                else str(mem.timestamp or ""),
                "importance": mem.importance,
            }
            for mem in memories
        ]

    async def get_formatted_history(self, limit: int | None = None) -> str:
        """Get formatted conversation history.

        Args:
            limit: Max messages

        Returns:
            Formatted history string
        """
        messages = await self.get_messages(limit)

        formatted = []
        for msg in messages:
            role = msg["role"].capitalize()
            formatted.append(f"{role}: {msg['content']}")

        return "\n\n".join(formatted)

    async def clear_history(self) -> None:
        """Clear conversation history."""
        memories = await self.memory_core.recall(
            query="", limit=1000, filters=RecallFilters(tags=[f"session:{self.session_id}"])
        )

        for mem in memories:
            await self.memory_core.forget(mem.memory_id)


class HaystackMemoryAdapter:
    """
    Convenience wrapper for creating Haystack-compatible memory.

    Example:
        ```python
        from mnemotree.integrations import HaystackMemoryAdapter

        adapter = HaystackMemoryAdapter(memory_core=memory_core)

        # Create memory for conversation
        chat_memory = adapter.create_chat_memory(session_id="conv_123")
        ```
    """

    def __init__(self, memory_core: MemoryCore):
        """Initialize the adapter.

        Args:
            memory_core: Mnemotree memory core instance
        """
        self.memory_core = memory_core

    def create_chat_memory(
        self,
        session_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_messages: int = 50,
    ) -> MnemotreeHaystackMemory:
        """Create memory instance for Haystack conversation.

        Args:
            session_id: Unique session identifier
            memory_type: Type of memory to store
            max_messages: Max messages to retrieve

        Returns:
            MnemotreeHaystackMemory instance
        """
        return MnemotreeHaystackMemory(
            memory_core=self.memory_core,
            session_id=session_id,
            memory_type=memory_type,
            max_messages=max_messages,
        )
