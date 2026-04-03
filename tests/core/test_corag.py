"""Unit tests for CoRAG (src/mnemotree/core/corag.py)."""
from __future__ import annotations

import pytest

from mnemotree.core.corag import ChainOfRetrieval, CoRAGConfig
from mnemotree.core.models import MemoryItem, MemoryType

# ---------------------------------------------------------------------------
# Helpers / Stubs
# ---------------------------------------------------------------------------


def _memory(mid: str, content: str, importance: float = 0.5) -> MemoryItem:
    return MemoryItem(
        memory_id=mid,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
    )


class StubRetriever:
    """Returns a fixed list of memories per query call."""

    def __init__(self, results_by_query: dict[str, list[MemoryItem]]) -> None:
        self.results = results_by_query
        self.calls: list[str] = []

    async def recall(self, query, *, limit=None, scoring=True, update_access=False):
        if not isinstance(query, str):
            return []
        self.calls.append(query)
        memories = self.results.get(query, [])
        if limit is not None:
            memories = memories[:limit]
        return memories


class StopHook:
    """Always returns None (stop immediately after first hop)."""

    async def reason_and_refine(self, original_query, hop_index, accumulated):
        return None


class OneHopHook:
    """Returns a second sub-query on the first hop, then stops."""

    def __init__(self, sub_query: str) -> None:
        self.sub_query = sub_query
        self.calls = 0

    async def reason_and_refine(self, original_query, hop_index, accumulated):
        if self.calls == 0:
            self.calls += 1
            return self.sub_query
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_stops_immediately_when_hook_returns_none():
    m1 = _memory("m1", "Alice knows Bob")
    retriever = StubRetriever({"Alice?": [m1]})
    hook = StopHook()
    chain = ChainOfRetrieval(retriever=retriever, hook=hook, cfg=CoRAGConfig(max_hops=4))
    memories, trace = await chain.retrieve("Alice?", limit=5)

    assert len(memories) >= 1
    assert "m1" in {m.memory_id for m in memories}
    assert trace == []  # hook returned None at hop 0, so no sub-query was added


@pytest.mark.asyncio
async def test_chain_executes_second_hop():
    m1 = _memory("m1", "Alice knows Bob")
    m2 = _memory("m2", "Bob works at ACME")

    retriever = StubRetriever({
        "Who does Alice know?": [m1],
        "Where does Bob work?": [m2],
    })
    hook = OneHopHook("Where does Bob work?")
    chain = ChainOfRetrieval(retriever=retriever, hook=hook, cfg=CoRAGConfig(max_hops=4))
    memories, trace = await chain.retrieve("Who does Alice know?", limit=10)

    ids = {m.memory_id for m in memories}
    assert "m1" in ids
    assert "m2" in ids
    assert "Where does Bob work?" in trace


@pytest.mark.asyncio
async def test_chain_stops_when_no_new_memories():
    m1 = _memory("m1", "Alice knows Bob")
    retriever = StubRetriever({
        "Alice?": [m1],
        "sub": [m1],  # same memory, no new additions
    })
    hook = OneHopHook("sub")
    cfg = CoRAGConfig(max_hops=4, min_new_memories=1)
    chain = ChainOfRetrieval(retriever=retriever, hook=hook, cfg=cfg)
    memories, trace = await chain.retrieve("Alice?", limit=5)

    # Should still return m1 even if second hop added nothing new
    assert len(memories) >= 1


@pytest.mark.asyncio
async def test_chain_respects_max_hops():
    """Hook always returns a sub-query; chain should stop at max_hops."""
    call_count = 0

    class InfiniteHook:
        async def reason_and_refine(self, original_query, hop_index, accumulated):
            nonlocal call_count
            call_count += 1
            return f"sub-{hop_index}"

    results = {f"sub-{i}": [_memory(f"m{i}", f"content {i}")] for i in range(10)}
    results["start"] = [_memory("m-start", "start content")]
    retriever = StubRetriever(results)
    chain = ChainOfRetrieval(
        retriever=retriever,
        hook=InfiniteHook(),
        cfg=CoRAGConfig(max_hops=3),
    )
    await chain.retrieve("start", limit=20)
    # hook should be called at most max_hops times
    assert call_count <= 3


@pytest.mark.asyncio
async def test_chain_returns_sorted_by_importance():
    low = _memory("low", "low importance", importance=0.2)
    high = _memory("high", "high importance", importance=0.9)
    retriever = StubRetriever({"q": [low, high]})
    chain = ChainOfRetrieval(retriever=retriever, hook=StopHook(), cfg=CoRAGConfig())
    memories, _ = await chain.retrieve("q", limit=10)
    assert memories[0].memory_id == "high"


@pytest.mark.asyncio
async def test_chain_deduplicates_memories():
    m1 = _memory("m1", "Alice")
    retriever = StubRetriever({
        "q": [m1],
        "follow-up": [m1],  # same memory returned again
    })
    hook = OneHopHook("follow-up")
    chain = ChainOfRetrieval(retriever=retriever, hook=hook, cfg=CoRAGConfig(max_hops=3))
    memories, _ = await chain.retrieve("q", limit=10)
    ids = [m.memory_id for m in memories]
    assert ids.count("m1") == 1  # no duplicates
