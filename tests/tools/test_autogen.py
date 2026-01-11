"""Tests for Autogen tool integration.

We intentionally patch `sys.modules["autogen"]` before importing the module under
test. This requires imports to occur after module-level setup.
"""

# ruff: noqa: E402

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock autogen module before importing
mock_autogen = MagicMock()
mock_autogen.register_function = lambda name=None, description=None: lambda f: f
sys.modules["autogen"] = mock_autogen

from mnemotree.core.memory import MemoryCore
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.tools.autogen import AutogenMemoryTool


@pytest.fixture
def mock_memory_core():
    core = AsyncMock(spec=MemoryCore)
    return core


@pytest.mark.asyncio
async def test_search_memories_scoring(mock_memory_core):
    # Setup
    tool = AutogenMemoryTool(mock_memory_core)

    # Mock data
    query_embedding = [0.1, 0.2, 0.3]
    memory_embedding = [0.1, 0.2, 0.3]  # Perfect match

    memory_item = MemoryItem(
        content="test content",
        memory_type=MemoryType.EPISODIC,
        embedding=memory_embedding,
        importance=0.5
    )

    # Mock output
    mock_memory_core.recall.return_value = [memory_item]
    mock_memory_core.get_embedding.return_value = query_embedding

    # Execute
    results = await tool.search_memories(query="test", limit=5, min_relevance=0.5)

    # Verify
    assert len(results) == 1
    assert results[0]["content"] == "test content"

    # Verify calls
    mock_memory_core.recall.assert_called_once()
    mock_memory_core.get_embedding.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_search_memories_scoring_filter(mock_memory_core):
    # Setup
    tool = AutogenMemoryTool(mock_memory_core)

    # Mock data - disjoint vectors (low score)
    query_embedding = [1.0, 0.0, 0.0]
    memory_embedding = [0.0, 1.0, 0.0]  # Orthogonal

    memory_item = MemoryItem(
        content="irrelevant",
        memory_type=MemoryType.EPISODIC,
        embedding=memory_embedding,
        importance=0.5
    )

    # Mock output
    mock_memory_core.recall.return_value = [memory_item]
    mock_memory_core.get_embedding.return_value = query_embedding

    # Execute - expect filtering due to min_relevance
    # Note: Score should be low due to orthogonal embeddings.
    results = await tool.search_memories(query="test", limit=5, min_relevance=0.9)

    # Verify
    assert len(results) == 0


@pytest.mark.asyncio
async def test_store_memory(mock_memory_core):
    tool = AutogenMemoryTool(mock_memory_core)

    mock_memory = MagicMock()
    mock_memory.memory_id = "mem-123"
    mock_memory_core.remember.return_value = mock_memory

    result = await tool.store_memory(content="test", importance=0.8, tags=["tag1"])

    assert result == "mem-123"
    mock_memory_core.remember.assert_called_once_with(
        "test", importance=0.8, tags=["tag1"]
    )
