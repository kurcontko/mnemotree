"""
LangGraph integration adapter for mnemotree.

Provides stateful memory for LangGraph workflows with persistence,
FSRS-4.5 decay, and semantic retrieval across graph executions.

Reference: https://langchain-ai.github.io/langgraph/
"""

from __future__ import annotations

from typing import Any

from ..core import MemoryCore, RecallFilters
from ..core.models import MemoryType


class MnemotreeLangGraphMemory:
    """
    LangGraph-compatible persistent memory using mnemotree.

    Provides:
    - Persistent memory across graph executions
    - Semantic search for relevant past states
    - FSRS-4.5 decay for state importance
    - Checkpoint integration

    Example:
        ```python
        from langgraph.graph import StateGraph
        from mnemotree import MemoryCore
        from mnemotree.store import ChromaMemoryStore
        from mnemotree.integrations import MnemotreeLangGraphMemory

        store = ChromaMemoryStore()
        memory_core = MemoryCore(store=store)

        # Create memory for graph
        graph_memory = MnemotreeLangGraphMemory(
            memory_core=memory_core,
            graph_id="workflow_1"
        )

        # Define graph
        workflow = StateGraph(AgentState)

        # Add memory-aware node
        async def research_node(state):
            # Retrieve relevant past research
            past_research = await graph_memory.retrieve_context(
                query=state["current_topic"],
                limit=5
            )
            # ... use past_research in processing
            return updated_state

        workflow.add_node("research", research_node)

        # Store results
        await graph_memory.store_state(state, node="research")
        ```
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        graph_id: str,
        memory_type: MemoryType = MemoryType.PROCEDURAL,
        max_context_items: int = 10,
    ):
        """Initialize LangGraph memory adapter.

        Args:
            memory_core: Mnemotree memory core instance
            graph_id: Unique identifier for this graph
            memory_type: Type of memory (procedural for workflows)
            max_context_items: Max items to retrieve for context
        """
        self.memory_core = memory_core
        self.graph_id = graph_id
        self.memory_type = memory_type
        self.max_context_items = max_context_items

    async def store_state(
        self,
        state: dict[str, Any],
        node: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store graph state at a specific node.

        Args:
            state: Current graph state
            node: Node name where state was captured
            metadata: Optional additional metadata
        """
        import json

        # Serialize state
        content = json.dumps(state, default=str)

        meta = metadata or {}
        meta.update({"node": node, "graph_id": self.graph_id})

        await self.memory_core.remember(
            content=content,
            memory_type=self.memory_type,
            tags=[
                f"graph:{self.graph_id}",
                f"node:{node}",
                "type:state",
            ],
            metadata=meta,
        )

    async def retrieve_context(
        self, query: str, node: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant past states.

        Args:
            query: Search query
            node: Optional node filter
            limit: Max results (uses max_context_items if None)

        Returns:
            List of past states
        """
        import json

        limit = limit or self.max_context_items

        tags = [f"graph:{self.graph_id}"]
        if node:
            tags.append(f"node:{node}")

        memories = await self.memory_core.recall(
            query=query, limit=limit, filters=RecallFilters(tags=tags)
        )

        states = []
        for mem in memories:
            try:
                state = json.loads(mem.content)
                states.append(state)
            except json.JSONDecodeError:
                # Skip malformed states
                continue

        return states

    async def store_decision(self, decision: str, reasoning: str, outcome: str, node: str) -> None:
        """Store a graph decision point.

        Args:
            decision: Decision made
            reasoning: Why the decision was made
            outcome: Result of the decision
            node: Node where decision occurred
        """
        content = f"Decision: {decision}\nReasoning: {reasoning}\nOutcome: {outcome}"

        await self.memory_core.remember(
            content=content,
            memory_type=MemoryType.EPISODIC,
            tags=[
                f"graph:{self.graph_id}",
                f"node:{node}",
                "type:decision",
            ],
            metadata={"decision": decision, "node": node},
        )

    async def get_execution_history(self, limit: int = 20) -> list[dict]:
        """Get past execution history for this graph.

        Args:
            limit: Max executions to retrieve

        Returns:
            List of execution records
        """
        memories = await self.memory_core.recall(
            query="",
            limit=limit,
            filters=RecallFilters(tags=[f"graph:{self.graph_id}"]),
        )

        return [
            {
                "content": mem.content,
                "timestamp": mem.timestamp,
                "node": (mem.metadata or {}).get("node"),
                "importance": mem.importance,
            }
            for mem in memories
        ]

    async def clear_history(self) -> None:
        """Clear all execution history for this graph."""
        memories = await self.memory_core.recall(
            query="", limit=1000, filters=RecallFilters(tags=[f"graph:{self.graph_id}"])
        )

        for mem in memories:
            await self.memory_core.forget(mem.memory_id)


class LangGraphMemoryAdapter:
    """
    Convenience wrapper for creating LangGraph-compatible memory.

    Example:
        ```python
        from mnemotree.integrations import LangGraphMemoryAdapter

        adapter = LangGraphMemoryAdapter(memory_core=memory_core)

        # Create memory for each graph
        workflow_memory = adapter.create_graph_memory(graph_id="workflow_1")
        pipeline_memory = adapter.create_graph_memory(graph_id="pipeline_1")
        ```
    """

    def __init__(self, memory_core: MemoryCore):
        """Initialize the adapter.

        Args:
            memory_core: Mnemotree memory core instance
        """
        self.memory_core = memory_core

    def create_graph_memory(
        self,
        graph_id: str,
        memory_type: MemoryType = MemoryType.PROCEDURAL,
        max_context_items: int = 10,
    ) -> MnemotreeLangGraphMemory:
        """Create memory instance for a LangGraph.

        Args:
            graph_id: Unique identifier for the graph
            memory_type: Type of memory to store
            max_context_items: Max items for context

        Returns:
            MnemotreeLangGraphMemory instance
        """
        return MnemotreeLangGraphMemory(
            memory_core=self.memory_core,
            graph_id=graph_id,
            memory_type=memory_type,
            max_context_items=max_context_items,
        )
