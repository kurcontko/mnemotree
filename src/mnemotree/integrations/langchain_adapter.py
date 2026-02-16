"""
LangChain integration adapter for mnemotree.

Provides BaseChatMessageHistory and BaseMemory implementations
that use mnemotree as the backend storage.
"""

from __future__ import annotations

from typing import Any

from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema.memory import BaseChatMessageHistory, BaseMemory

from ..core import MemoryCore, RecallFilters
from ..core.models import MemoryType


class MnemotreeChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by mnemotree.

    Stores conversation messages as episodic memories with full
    mnemotree features: decay, importance scoring, semantic search.

    Example:
        ```python
        from mnemotree import MemoryCore
        from mnemotree.store import ChromaMemoryStore
        from mnemotree.integrations import LangChainMemoryAdapter

        store = ChromaMemoryStore()
        memory_core = MemoryCore(store=store)

        # Use with LangChain
        history = MnemotreeChatMessageHistory(
            memory_core=memory_core,
            session_id="conversation-123"
        )

        history.add_user_message("What's the weather?")
        history.add_ai_message("It's sunny today!")

        messages = history.messages  # Retrieve with decay applied
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        session_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_messages: int = 50,
    ):
        """Initialize the message history.

        Args:
            memory_core: Mnemotree memory core instance
            session_id: Unique identifier for this conversation
            memory_type: Type of memory to store messages as
            max_messages: Maximum number of recent messages to retrieve
        """
        self.memory_core = memory_core
        self.session_id = session_id
        self.memory_type = memory_type
        self.max_messages = max_messages

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve all messages for this session with decay applied."""
        import asyncio

        # Run async recall in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, can't use run_until_complete
            # Fall back to empty list (caller should use async methods)
            return []

        memories = loop.run_until_complete(
            self.memory_core.recall(
                query="",  # Empty query to get all
                limit=self.max_messages,
                filters=RecallFilters(tags=[f"session:{self.session_id}"]),
            )
        )

        messages = []
        for mem in memories:
            # Parse message type from metadata
            msg_type = mem.metadata.get("message_type", "human")
            if msg_type == "ai":
                messages.append(AIMessage(content=mem.content))
            else:
                messages.append(HumanMessage(content=mem.content))

        return messages

    async def aget_messages(self) -> list[BaseMessage]:
        """Async version of messages property."""
        memories = await self.memory_core.recall(
            query="",
            limit=self.max_messages,
            filters=RecallFilters(tags=[f"session:{self.session_id}"]),
        )

        messages = []
        for mem in memories:
            msg_type = mem.metadata.get("message_type", "human")
            if msg_type == "ai":
                messages.append(AIMessage(content=mem.content))
            else:
                messages.append(HumanMessage(content=mem.content))

        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        import asyncio

        msg_type = "ai" if isinstance(message, AIMessage) else "human"

        loop = asyncio.get_event_loop()
        # Convert message.content to string if it's not already
        content_str = (
            str(message.content) if not isinstance(message.content, str) else message.content
        )

        loop.run_until_complete(
            self.memory_core.remember(
                content=content_str,
                memory_type=self.memory_type,
                tags=[f"session:{self.session_id}", "message"],
                metadata={"message_type": msg_type},
            )
        )

    async def aadd_message(self, message: BaseMessage) -> None:
        """Async version of add_message."""
        msg_type = "ai" if isinstance(message, AIMessage) else "human"

        # Convert message.content to string if it's not already
        content_str = (
            str(message.content) if not isinstance(message.content, str) else message.content
        )

        await self.memory_core.remember(
            content=content_str,
            memory_type=self.memory_type,
            tags=[f"session:{self.session_id}", "message"],
            metadata={"message_type": msg_type},
        )

    def clear(self) -> None:
        """Clear all messages for this session."""
        import asyncio

        loop = asyncio.get_event_loop()
        # Note: MemoryCore doesn't have delete_by_filter yet
        # This is a placeholder - implement in MemoryCore
        loop.run_until_complete(self._clear_session())

    async def _clear_session(self) -> None:
        """Internal async clear implementation."""
        # Get all messages for this session
        memories = await self.memory_core.recall(
            query="",
            limit=1000,
            filters=RecallFilters(tags=[f"session:{self.session_id}"]),
        )

        # Delete each one
        for mem in memories:
            await self.memory_core.forget(mem.memory_id)


class LangChainMemoryAdapter(BaseMemory):
    """
    LangChain memory adapter that provides semantic retrieval.

    Unlike standard LangChain memory which just buffers recent messages,
    this adapter:
    - Performs semantic search to find relevant past interactions
    - Applies importance decay over time
    - Supports memory consolidation

    Example:
        ```python
        from langchain.chains import ConversationChain
        from mnemotree.integrations.langchain_adapter import LangChainMemoryAdapter

        memory = LangChainMemoryAdapter(
            memory_core=memory_core,
            session_id="user-123",
            return_messages=True
        )

        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        # Conversation will automatically retrieve semantically relevant memories
        response = conversation.predict(input="What did I say about Python?")
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        session_id: str,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        max_relevant: int = 5,
    ):
        """Initialize the memory adapter.

        Args:
            memory_core: Mnemotree memory core instance
            session_id: Unique identifier for this conversation
            memory_key: Key to use for memory in prompt
            input_key: Key for input in conversation dict
            output_key: Key for output in conversation dict
            return_messages: Whether to return Message objects or strings
            max_relevant: Maximum number of relevant memories to retrieve
        """
        self.memory_core = memory_core
        self.session_id = session_id
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.max_relevant = max_relevant

    @property
    def memory_variables(self) -> list[str]:
        """Return list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load memory variables using semantic search on input."""
        import asyncio

        query = inputs.get(self.input_key, "")

        loop = asyncio.get_event_loop()
        if loop.is_running():
            return {self.memory_key: ""}

        relevant = loop.run_until_complete(
            self.memory_core.recall(
                query=query,
                limit=self.max_relevant,
                filters=RecallFilters(tags=[f"session:{self.session_id}"]),
            )
        )

        if self.return_messages:
            messages = []
            for mem in relevant:
                msg_type = mem.metadata.get("message_type", "human")
                if msg_type == "ai":
                    messages.append(AIMessage(content=mem.content))
                else:
                    messages.append(HumanMessage(content=mem.content))
            return {self.memory_key: messages}
        else:
            context = "\n".join([mem.to_str_llm() for mem in relevant])
            return {self.memory_key: context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save context to mnemotree."""
        import asyncio

        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        loop = asyncio.get_event_loop()

        # Save user message
        loop.run_until_complete(
            self.memory_core.remember(
                content=user_input,
                memory_type=MemoryType.EPISODIC,
                tags=[f"session:{self.session_id}", "user_message"],
                metadata={"message_type": "human"},
            )
        )

        # Save AI response
        loop.run_until_complete(
            self.memory_core.remember(
                content=ai_output,
                memory_type=MemoryType.EPISODIC,
                tags=[f"session:{self.session_id}", "ai_message"],
                metadata={"message_type": "ai"},
            )
        )

    def clear(self) -> None:
        """Clear all memories for this session."""
        import asyncio

        loop = asyncio.get_event_loop()

        async def _clear():
            memories = await self.memory_core.recall(
                query="",
                limit=1000,
                filters=RecallFilters(tags=[f"session:{self.session_id}"]),
            )
            for mem in memories:
                await self.memory_core.forget(mem.memory_id)

        loop.run_until_complete(_clear())
