"""Tests for MemoryClusterer — focus on edge cases with missing embeddings."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mnemotree.analysis.clustering import MemoryClusterer
from mnemotree.core.models import MemoryItem, MemoryType


def _mem(mid: str, content: str, embedding: list[float] | None = None) -> MemoryItem:
    return MemoryItem(
        memory_id=mid,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=embedding,
    )


@pytest.fixture
def clusterer():
    summarizer = AsyncMock()
    summarizer.summarize = AsyncMock(return_value="cluster summary")
    return MemoryClusterer(summarizer)


@pytest.mark.asyncio
async def test_cluster_empty_list(clusterer):
    """Empty input returns empty ClusteringResult."""
    result = await clusterer.cluster_memories([])
    assert result.cluster_ids == []
    assert result.cluster_sizes == {}


@pytest.mark.asyncio
async def test_cluster_all_missing_embeddings(clusterer):
    """All memories without embeddings returns empty result."""
    memories = [_mem("a", "foo"), _mem("b", "bar")]
    result = await clusterer.cluster_memories(memories, method="vector_dbscan")
    assert result.cluster_ids == []


@pytest.mark.asyncio
async def test_cluster_mixed_embeddings_no_index_error(clusterer):
    """Memories with and without embeddings should NOT raise IndexError.

    This was a bug: cluster_ids aligned with valid subset, but code
    iterated over all memories causing IndexError.
    """
    dim = 8
    memories = [
        _mem("a", "has embedding", embedding=[0.1] * dim),
        _mem("b", "no embedding"),  # No embedding
        _mem("c", "also has embedding", embedding=[0.2] * dim),
        _mem("d", "another no embedding"),  # No embedding
        _mem("e", "third with embedding", embedding=[0.3] * dim),
    ]
    # Use DBSCAN with very loose eps so all valid memories land in one cluster
    result = await clusterer.cluster_memories(
        memories, method="vector_dbscan", eps=2.0, min_samples=1
    )
    # Should succeed without IndexError
    assert len(result.cluster_ids) > 0
    assert len(result.cluster_sizes) > 0
