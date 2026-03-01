"""
CrewAI integration adapter for mnemotree.

Provides memory capabilities for CrewAI multi-agent teams with
role-based memory, FSRS-4.5 decay, and semantic retrieval.

Reference: https://www.crewai.com/
"""

from __future__ import annotations

from typing import Any

from ..core import MemoryCore, RecallFilters
from ..core.models import MemoryType


class MnemotreeCrewMemory:
    """
    CrewAI-compatible memory using mnemotree backend.

    Provides role-specific memory for CrewAI agents with:
    - Per-agent and crew-wide memory
    - Semantic search across agent interactions
    - FSRS-4.5 decay for importance
    - Task-specific memory tracking

    Example:
        ```python
        from crewai import Agent, Task, Crew
        from mnemotree import MemoryCore
        from mnemotree.store import ChromaMemoryStore
        from mnemotree.integrations import MnemotreeCrewMemory

        store = ChromaMemoryStore()
        memory_core = MemoryCore(store=store)

        # Create memory for crew
        crew_memory = MnemotreeCrewMemory(
            memory_core=memory_core,
            crew_id="research_team"
        )

        # Create agents with shared memory
        researcher = Agent(
            role="Researcher",
            goal="Find relevant information",
            backstory="Expert researcher",
            memory=crew_memory.get_agent_memory("researcher")
        )

        writer = Agent(
            role="Writer",
            goal="Write clear summaries",
            backstory="Professional writer",
            memory=crew_memory.get_agent_memory("writer")
        )

        # Agents can access each other's memories
        crew = Crew(agents=[researcher, writer])
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        crew_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_context_items: int = 10,
    ):
        """Initialize CrewAI memory adapter.

        Args:
            memory_core: Mnemotree memory core instance
            crew_id: Unique identifier for this crew
            memory_type: Type of memory to store
            max_context_items: Max items to retrieve for context
        """
        self.memory_core = memory_core
        self.crew_id = crew_id
        self.memory_type = memory_type
        self.max_context_items = max_context_items
        self._agent_memories: dict[str, AgentMemory] = {}

    def get_agent_memory(self, agent_role: str) -> AgentMemory:
        """Get or create memory for a specific agent role.

        Args:
            agent_role: Role identifier for the agent (e.g., "researcher", "writer")

        Returns:
            AgentMemory instance for this agent
        """
        if agent_role not in self._agent_memories:
            self._agent_memories[agent_role] = AgentMemory(
                memory_core=self.memory_core,
                crew_id=self.crew_id,
                agent_role=agent_role,
                memory_type=self.memory_type,
                max_context_items=self.max_context_items,
            )
        return self._agent_memories[agent_role]

    async def store_task_output(self, task_description: str, output: str, agent_role: str) -> None:
        """Store task execution output.

        Args:
            task_description: Description of the task
            output: Task output/result
            agent_role: Role of agent that executed the task
        """
        await self.memory_core.remember(
            content=f"Task: {task_description}\nOutput: {output}",
            memory_type=MemoryType.PROCEDURAL,  # Tasks are procedural
            tags=[
                f"crew:{self.crew_id}",
                f"agent:{agent_role}",
                "type:task",
            ],
            metadata={"task": task_description, "agent_role": agent_role},
        )

    async def get_crew_context(self, query: str, limit: int = 10) -> str:
        """Get relevant context across all crew members.

        Args:
            query: Search query
            limit: Maximum items to retrieve

        Returns:
            Formatted context string
        """
        memories = await self.memory_core.recall(
            query=query, limit=limit, filters=RecallFilters(tags=[f"crew:{self.crew_id}"])
        )

        context_parts = []
        for mem in memories:
            agent = (mem.metadata or {}).get("agent_role", "unknown")
            context_parts.append(f"[{agent}]: {mem.content}")

        return "\n\n".join(context_parts)

    async def clear_crew_memory(self) -> None:
        """Clear all memory for this crew."""
        memories = await self.memory_core.recall(
            query="", limit=1000, filters=RecallFilters(tags=[f"crew:{self.crew_id}"])
        )

        for mem in memories:
            await self.memory_core.forget(mem.memory_id)


class AgentMemory:
    """Memory interface for individual CrewAI agents."""

    def __init__(
        self,
        memory_core: MemoryCore,
        crew_id: str,
        agent_role: str,
        memory_type: MemoryType,
        max_context_items: int,
    ):
        """Initialize agent memory.

        Args:
            memory_core: Mnemotree memory core
            crew_id: Crew identifier
            agent_role: Agent role identifier
            memory_type: Type of memory to store
            max_context_items: Max context items
        """
        self.memory_core = memory_core
        self.crew_id = crew_id
        self.agent_role = agent_role
        self.memory_type = memory_type
        self.max_context_items = max_context_items

    async def store(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Store information in memory.

        Args:
            content: Content to store
            metadata: Optional metadata
        """
        meta = metadata or {}
        meta["agent_role"] = self.agent_role

        await self.memory_core.remember(
            content=content,
            memory_type=self.memory_type,
            tags=[
                f"crew:{self.crew_id}",
                f"agent:{self.agent_role}",
            ],
            metadata=meta,
        )

    async def retrieve(self, query: str, limit: int | None = None) -> list[str]:
        """Retrieve relevant memories.

        Args:
            query: Search query
            limit: Max results (uses max_context_items if None)

        Returns:
            List of relevant memory contents
        """
        limit = limit or self.max_context_items

        memories = await self.memory_core.recall(
            query=query,
            limit=limit,
            filters=RecallFilters(tags=[f"crew:{self.crew_id}", f"agent:{self.agent_role}"]),
        )

        return [mem.content for mem in memories]

    async def get_context(self, query: str) -> str:
        """Get formatted context for this agent.

        Args:
            query: Search query

        Returns:
            Formatted context string
        """
        contents = await self.retrieve(query)
        return "\n".join(contents)


class CrewAIMemoryAdapter:
    """
    Convenience wrapper for creating CrewAI-compatible memory.

    Example:
        ```python
        from mnemotree.integrations import CrewAIMemoryAdapter

        adapter = CrewAIMemoryAdapter(memory_core=memory_core)

        # Create memory for crew
        crew_memory = adapter.create_crew_memory(crew_id="research_team")

        # Get agent-specific memory
        researcher_memory = crew_memory.get_agent_memory("researcher")
        ```
    """

    def __init__(self, memory_core: MemoryCore):
        """Initialize the adapter.

        Args:
            memory_core: Mnemotree memory core instance
        """
        self.memory_core = memory_core

    def create_crew_memory(
        self,
        crew_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        max_context_items: int = 10,
    ) -> MnemotreeCrewMemory:
        """Create memory instance for a CrewAI crew.

        Args:
            crew_id: Unique identifier for the crew
            memory_type: Type of memory to store
            max_context_items: Max items for context

        Returns:
            MnemotreeCrewMemory instance
        """
        return MnemotreeCrewMemory(
            memory_core=self.memory_core,
            crew_id=crew_id,
            memory_type=memory_type,
            max_context_items=max_context_items,
        )
