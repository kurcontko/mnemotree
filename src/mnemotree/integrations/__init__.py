"""
Framework integrations for mnemotree.

Provides adapters for popular agent frameworks:
- LangChain: BaseChatMessageHistory and BaseMemory implementations
- LlamaIndex: Vector memory for chat engines
- AutoGen: Multi-agent conversation memory
- CrewAI: Role-based team memory
- LangGraph: Stateful workflow memory
- Haystack: RAG conversational memory

Each adapter provides:
- Semantic search over historical interactions
- FSRS-4.5 importance decay
- Framework-specific interfaces
- Async support where applicable
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .autogen_adapter import AutoGenMemoryAdapter, MnemotreeAutoGenMemory
    from .crewai_adapter import CrewAIMemoryAdapter, MnemotreeCrewMemory
    from .haystack_adapter import HaystackMemoryAdapter, MnemotreeHaystackMemory
    from .langchain_adapter import LangChainMemoryAdapter, MnemotreeChatMessageHistory
    from .langgraph_adapter import LangGraphMemoryAdapter, MnemotreeLangGraphMemory
    from .llamaindex_adapter import LlamaIndexMemoryAdapter, MnemotreeVectorMemory

__all__ = [
    # LangChain
    "LangChainMemoryAdapter",
    "MnemotreeChatMessageHistory",
    # LlamaIndex
    "LlamaIndexMemoryAdapter",
    "MnemotreeVectorMemory",
    # AutoGen
    "AutoGenMemoryAdapter",
    "MnemotreeAutoGenMemory",
    # CrewAI
    "CrewAIMemoryAdapter",
    "MnemotreeCrewMemory",
    # LangGraph
    "LangGraphMemoryAdapter",
    "MnemotreeLangGraphMemory",
    # Haystack
    "HaystackMemoryAdapter",
    "MnemotreeHaystackMemory",
]
