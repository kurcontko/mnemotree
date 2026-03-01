"""Tests for ConflictDetector."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core._internal.conflict import ConflictDetector
from mnemotree.core.models import MemoryItem, MemoryType


def _make_memory(
    memory_id: str,
    content: str,
    entities: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        entities=entities or {},
    )


@pytest.mark.asyncio
async def test_no_conflict_when_no_shared_entities():
    detector = ConflictDetector(llm=None)
    new_mem = _make_memory("new", "Paris is the capital of France.", {"Paris": "city"})
    candidate = _make_memory("c1", "Rome is the capital of Italy.", {"Rome": "city"})

    results = await detector.detect(new_mem, [candidate])

    assert results == []


@pytest.mark.asyncio
async def test_no_conflict_when_no_negation():
    detector = ConflictDetector(llm=None)
    new_mem = _make_memory("new", "Python is dynamic.", {"Python": "language"})
    candidate = _make_memory("c1", "Python is interpreted.", {"Python": "language"})

    results = await detector.detect(new_mem, [candidate])

    assert results == []


@pytest.mark.asyncio
async def test_heuristic_conflict_detected():
    detector = ConflictDetector(llm=None)
    new_mem = _make_memory("new", "Python is not compiled.", {"Python": "language"})
    candidate = _make_memory("c1", "Python is a compiled language.", {"Python": "language"})

    results = await detector.detect(new_mem, [candidate])

    assert len(results) == 1
    conflict_mem, reason = results[0]
    assert conflict_mem.memory_id == "c1"
    assert "heuristic" in reason


@pytest.mark.asyncio
async def test_llm_conflict_skips_on_no_contradiction():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(content='{"contradicts": false, "reason": ""}')
    )
    detector = ConflictDetector(llm=llm)
    new_mem = _make_memory("new", "Python is not compiled.", {"Python": "language"})
    candidate = _make_memory("c1", "Python is a compiled language.", {"Python": "language"})

    results = await detector.detect(new_mem, [candidate])

    # LLM says no contradiction → empty results
    assert results == []


@pytest.mark.asyncio
async def test_llm_conflict_confirms_contradiction():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"contradicts": true, "reason": "directly oppose each other"}'
        )
    )
    detector = ConflictDetector(llm=llm)
    new_mem = _make_memory("new", "Python is not compiled.", {"Python": "language"})
    candidate = _make_memory("c1", "Python is a compiled language.", {"Python": "language"})

    results = await detector.detect(new_mem, [candidate])

    assert len(results) == 1
    _, reason = results[0]
    assert "directly oppose" in reason


@pytest.mark.asyncio
async def test_llm_failure_falls_back_gracefully():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
    detector = ConflictDetector(llm=llm)
    new_mem = _make_memory("new", "Python is not compiled.", {"Python": "language"})
    candidate = _make_memory("c1", "Python is a compiled language.", {"Python": "language"})

    # Should not raise; LLM failure → treated as no contradiction
    results = await detector.detect(new_mem, [candidate])

    assert results == []
