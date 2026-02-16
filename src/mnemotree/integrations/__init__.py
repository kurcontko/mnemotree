"""
Framework integrations for mnemotree.

Provides adapters for popular agent frameworks:
- LangChain (langchain_adapter.py)
- LlamaIndex (llamaindex_adapter.py)
- AutoGen (autogen_adapter.py)
- CrewAI (crewai_adapter.py)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .langchain_adapter import LangChainMemoryAdapter
    from .llamaindex_adapter import LlamaIndexMemoryAdapter

__all__ = [
    "LangChainMemoryAdapter",
    "LlamaIndexMemoryAdapter",
]
