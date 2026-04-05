"""
AutoGen integration adapter for mnemotree.

Provides memory capabilities for Microsoft AutoGen multi-agent conversations
with FSRS-4.5 decay and semantic retrieval.

Reference: https://microsoft.github.io/autogen/
"""

from __future__ import annotations

from typing import Any

from ..core import MemoryCore, RecallFilters
from ..core.models import MemoryType


class MnemotreeAutoGenMemory:
    """
    AutoGen-compatible memory using mnemotree backend.

    Integrates with AutoGen's ConversableAgent to provide:
    - Long-term memory across conversations
    - Semantic search for relevant context
    - FSRS-4.5 importance decay
    - Multi-agent conversation tracking

    Example:
        ```python
        from autogen import ConversableAgent, UserProxyAgent
        from mnemotree import MemoryCore
        from mnemotree.store import ChromaMemoryStore
        from mnemotree.integrations import MnemotreeAutoGenMemory

        store = ChromaMemoryStore()
        memory_core = MemoryCore(store=store)

        # Create memory for agent
        agent_memory = MnemotreeAutoGenMemory(
            memory_core=memory_core,
            agent_id="research_assistant"
        )

        # Create AutoGen agent with custom memory
        assistant = ConversableAgent(
            name="assistant",
            llm_config={"model": "gpt-4"},
            system_message="You are a helpful research assistant."
        )

        # Add memory hooks
        agent_memory.register_hooks(assistant)

        # Conversations are now automatically stored
        user = UserProxyAgent(name="user")
        user.initiate_chat(assistant, message="What did we discuss about AI?")
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        agent_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_context_messages: int = 10,
    ):
        """Initialize AutoGen memory adapter.

        Args:
            memory_core: Mnemotree memory core instance
            agent_id: Unique identifier for this agent
            memory_type: Type of memory to store messages as
            max_context_messages: Max messages to retrieve for context
        """
        self.memory_core = memory_core
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.max_context_messages = max_context_messages

    def register_hooks(self, agent: Any) -> None:
        """Register hooks to AutoGen agent for automatic memory storage.

        Args:
            agent: AutoGen ConversableAgent instance
        """
        # Store original reply functions
        original_generate_reply = agent.generate_reply

        async def memory_enhanced_reply(messages: list[dict] | None = None, **kwargs):
            """Wrap reply generation with memory retrieval."""
            if messages:
                # Get last message as query
                last_msg = messages[-1]
                query = last_msg.get("content", "")

                # Retrieve relevant memories
                relevant = await self.memory_core.recall(
                    query=query,
                    limit=self.max_context_messages,
                    filters=RecallFilters(tags=[f"agent:{self.agent_id}"]),
                )

                # Inject memory context
                if relevant:
                    memory_context = "\n".join(
                        [
                            f"[Memory from {mem.timestamp.strftime('%Y-%m-%d') if hasattr(mem.timestamp, 'strftime') else str(mem.timestamp or '')}]: {mem.content}"
                            for mem in relevant
                        ]
                    )
                    # Prepend memory to system message or first user message
                    enhanced_messages = [
                        {
                            "role": "system",
                            "content": f"Relevant past context:\n{memory_context}",
                        }
                    ] + messages
                else:
                    enhanced_messages = messages

                # Generate reply with enhanced context
                reply = await original_generate_reply(enhanced_messages, **kwargs)

                # Store this interaction
                if reply:
                    await self.store_message(query, str(reply))

                return reply
            else:
                return await original_generate_reply(messages, **kwargs)

        # Replace with memory-enhanced version
        agent.generate_reply = memory_enhanced_reply

    async def store_message(self, user_message: str, agent_reply: str) -> None:
        """Store a conversation turn in memory.

        Args:
            user_message: User's message
            agent_reply: Agent's response
        """
        # Store user message
        await self.memory_core.remember(
            content=user_message,
            memory_type=self.memory_type,
            tags=[f"agent:{self.agent_id}", "role:user"],
            metadata={"role": "user"},
        )

        # Store agent reply
        await self.memory_core.remember(
            content=agent_reply,
            memory_type=self.memory_type,
            tags=[f"agent:{self.agent_id}", "role:assistant"],
            metadata={"role": "assistant"},
        )

    async def get_conversation_history(self, limit: int = 50) -> list[dict]:
        """Retrieve conversation history for this agent.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dicts with role and content
        """
        memories = await self.memory_core.recall(
            query="",
            limit=limit,
            filters=RecallFilters(tags=[f"agent:{self.agent_id}"]),
        )

        return [
            {"role": (mem.metadata or {}).get("role", "user"), "content": mem.content}
            for mem in memories
        ]

    async def search_context(self, query: str, limit: int = 5) -> list[str]:
        """Search for relevant context from past conversations.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of relevant message contents
        """
        memories = await self.memory_core.recall(
            query=query, limit=limit, filters=RecallFilters(tags=[f"agent:{self.agent_id}"])
        )

        return [mem.content for mem in memories]

    async def clear_history(self) -> None:
        """Clear all conversation history for this agent."""
        memories = await self.memory_core.recall(
            query="", limit=1000, filters=RecallFilters(tags=[f"agent:{self.agent_id}"])
        )

        for mem in memories:
            await self.memory_core.forget(mem.memory_id)


class AutoGenMemoryAdapter:
    """
    Convenience wrapper for creating AutoGen-compatible memory.

    Example:
        ```python
        from mnemotree.integrations import AutoGenMemoryAdapter

        adapter = AutoGenMemoryAdapter(memory_core=memory_core)

        # Create memory for each agent
        assistant_memory = adapter.create_agent_memory(agent_id="assistant")
        researcher_memory = adapter.create_agent_memory(agent_id="researcher")
        ```
    """

    def __init__(self, memory_core: MemoryCore):
        """Initialize the adapter.

        Args:
            memory_core: Mnemotree memory core instance
        """
        self.memory_core = memory_core

    def create_agent_memory(
        self,
        agent_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_context_messages: int = 10,
    ) -> MnemotreeAutoGenMemory:
        """Create memory instance for an AutoGen agent.

        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to store
            max_context_messages: Max messages for context

        Returns:
            MnemotreeAutoGenMemory instance
        """
        return MnemotreeAutoGenMemory(
            memory_core=self.memory_core,
            agent_id=agent_id,
            memory_type=memory_type,
            max_context_messages=max_context_messages,
        )
