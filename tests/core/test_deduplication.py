"""Tests for DedupChecker."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core._internal.deduplication import DedupChecker
from mnemotree.core.models import MemoryItem, MemoryType


def _make_memory(memory_id: str, embedding: list[float], user_id: str | None = None) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content="test content",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=embedding,
        user_id=user_id,
    )


@pytest.mark.asyncio
async def test_find_duplicate_returns_none_when_no_candidates():
    retrieval = MagicMock()
    retrieval.recall = AsyncMock(return_value=[])
    checker = DedupChecker(retrieval, threshold=0.88)

    result = await checker.find_duplicate([1.0, 0.0], "query text", user_id=None, limit=3)

    assert result is None


@pytest.mark.asyncio
async def test_find_duplicate_returns_none_when_below_threshold():
    # Orthogonal vectors → cosine = 0.0, well below threshold
    candidate = _make_memory("m1", [0.0, 1.0])
    retrieval = MagicMock()
    retrieval.recall = AsyncMock(return_value=[candidate])
    checker = DedupChecker(retrieval, threshold=0.88)

    result = await checker.find_duplicate([1.0, 0.0], "query text", user_id=None, limit=3)

    assert result is None


@pytest.mark.asyncio
async def test_find_duplicate_returns_match_above_threshold():
    # Identical vectors → cosine = 1.0
    vec = [1.0, 0.0]
    candidate = _make_memory("m1", vec)
    retrieval = MagicMock()
    retrieval.recall = AsyncMock(return_value=[candidate])
    checker = DedupChecker(retrieval, threshold=0.88)

    result = await checker.find_duplicate(vec, "query text", user_id=None, limit=3)

    assert result is not None
    found_memory, score = result
    assert found_memory.memory_id == "m1"
    assert score >= 0.88


@pytest.mark.asyncio
async def test_find_duplicate_filters_by_user_id():
    vec = [1.0, 0.0]
    candidate = _make_memory("m1", vec, user_id="alice")
    retrieval = MagicMock()
    retrieval.recall = AsyncMock(return_value=[candidate])
    checker = DedupChecker(retrieval, threshold=0.88)

    # Different user_id — should not match
    result = await checker.find_duplicate(vec, "query text", user_id="bob", limit=3)

    assert result is None


@pytest.mark.asyncio
async def test_find_duplicate_picks_best_match():
    vec = [1.0, 0.0]

    # Two candidates: one exactly matches (score=1.0), one is slightly off
    good = _make_memory("good", [1.0, 0.0])
    ok = _make_memory("ok", [0.95, 0.31])  # cosine ≈ 0.95 but not 1.0
    retrieval = MagicMock()
    retrieval.recall = AsyncMock(return_value=[ok, good])
    checker = DedupChecker(retrieval, threshold=0.88)

    result = await checker.find_duplicate(vec, "query text", user_id=None, limit=3)

    assert result is not None
    assert result[0].memory_id == "good"


@pytest.mark.asyncio
async def test_find_duplicate_skips_candidates_without_embedding():
    candidate = _make_memory("m1", [])
    candidate = candidate.model_copy(update={"embedding": None})
    retrieval = MagicMock()
    retrieval.recall = AsyncMock(return_value=[candidate])
    checker = DedupChecker(retrieval, threshold=0.88)

    result = await checker.find_duplicate([1.0, 0.0], "query text", user_id=None, limit=3)

    assert result is None
