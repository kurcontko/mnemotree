"""Tests for core internal modules: conflict, deduplication, enrichment, fact_decomposer, ingestion_queue."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core._internal.conflict import ConflictDetector
from mnemotree.core._internal.deduplication import DedupChecker
from mnemotree.core._internal.ingestion_queue import IngestionRequest, MemoryIngestionQueue
from mnemotree.core.models import MemoryItem, MemoryType


def _mem(mid: str, content: str, **kwargs) -> MemoryItem:
    """Helper to create MemoryItem with required fields."""
    defaults = {"memory_type": MemoryType.EPISODIC, "importance": 0.5}
    defaults.update(kwargs)
    return MemoryItem(memory_id=mid, content=content, **defaults)


# --- ConflictDetector ---


class TestConflictDetector:
    def test_heuristic_conflict_empty_entities(self):
        """Heuristic conflict check should handle empty entities gracefully."""
        detector = ConflictDetector()
        a = _mem("a", "X is not good", entities={})
        b = _mem("b", "X is good", entities={})
        assert detector._heuristic_conflict(a, b) is False

    def test_heuristic_conflict_shared_entities_with_negation(self):
        """Should detect conflict when shared entities and negation words present."""
        detector = ConflictDetector()
        a = _mem("a", "X is not good", entities={"X": "thing"})
        b = _mem("b", "X is good", entities={"X": "thing"})
        assert detector._heuristic_conflict(a, b) is True

    def test_heuristic_conflict_shared_entities_no_negation(self):
        """Should not detect conflict when no negation words."""
        detector = ConflictDetector()
        a = _mem("a", "X is great", entities={"X": "thing"})
        b = _mem("b", "X is wonderful", entities={"X": "thing"})
        assert detector._heuristic_conflict(a, b) is False

    def test_heuristic_conflict_no_shared_entities(self):
        """Should not detect conflict when no shared entities."""
        detector = ConflictDetector()
        a = _mem("a", "X is not good", entities={"X": "thing"})
        b = _mem("b", "Y is not good", entities={"Y": "thing"})
        assert detector._heuristic_conflict(a, b) is False

    def test_heuristic_conflict_defensive_none_entities(self):
        """Even with entities forced to None via model_construct, should not crash."""
        detector = ConflictDetector()
        # Simulate corrupt data from DB deserialization
        a = MemoryItem.model_construct(
            memory_id="a",
            content="X is not good",
            entities=None,
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
        )
        b = MemoryItem.model_construct(
            memory_id="b",
            content="X is good",
            entities=None,
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
        )
        # Should not crash
        assert detector._heuristic_conflict(a, b) is False

    @pytest.mark.asyncio
    async def test_detect_no_llm(self):
        """Detect should work without LLM using heuristics only."""
        detector = ConflictDetector(llm=None)
        new = _mem("new", "X is not good", entities={"X": "thing"})
        existing = _mem("old", "X is good", entities={"X": "thing"})
        results = await detector.detect(new, [existing])
        assert len(results) == 1
        assert results[0][0].memory_id == "old"
        assert "heuristic" in results[0][1]

    @pytest.mark.asyncio
    async def test_detect_empty_candidates(self):
        """Detect with no candidates returns empty."""
        detector = ConflictDetector()
        new = _mem("new", "hello")
        results = await detector.detect(new, [])
        assert results == []


# --- DedupChecker ---


class TestDedupChecker:
    @pytest.mark.asyncio
    async def test_find_duplicate_none_embedding(self):
        """Should return None immediately if embedding is None."""
        retrieval = MagicMock()
        checker = DedupChecker(retrieval=retrieval, threshold=0.9)
        result = await checker.find_duplicate(embedding=None, content="test", user_id=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_duplicate_empty_embedding(self):
        """Should return None immediately if embedding is empty list."""
        retrieval = MagicMock()
        checker = DedupChecker(retrieval=retrieval, threshold=0.9)
        result = await checker.find_duplicate(embedding=[], content="test", user_id=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_duplicate_no_candidates(self):
        """Should return None when no candidates match."""
        retrieval = AsyncMock()
        retrieval.recall = AsyncMock(return_value=[])
        checker = DedupChecker(retrieval=retrieval, threshold=0.9)
        result = await checker.find_duplicate(
            embedding=[1.0, 0.0, 0.0], content="test", user_id=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_find_duplicate_skips_none_embedding_candidates(self):
        """Should skip candidates without embeddings."""
        candidate = _mem("c1", "test")  # No embedding set
        retrieval = AsyncMock()
        retrieval.recall = AsyncMock(return_value=[candidate])
        checker = DedupChecker(retrieval=retrieval, threshold=0.5)
        result = await checker.find_duplicate(
            embedding=[1.0, 0.0, 0.0], content="test", user_id=None
        )
        assert result is None


# --- IngestionQueue ---


class TestIngestionQueue:
    def _make_request(self, mid: str) -> IngestionRequest:
        return IngestionRequest(
            memory_id=mid,
            content="hello",
            memory_type=None,
            importance=None,
            tags=None,
            context=None,
            analyze=None,
            summarize=None,
            references=None,
            timestamp=None,
            source=None,
            author=None,
            metadata=None,
            conversation_id=None,
            user_id=None,
        )

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        """Queue should process enqueued items."""
        processed = []

        async def handler(req: IngestionRequest) -> None:
            processed.append(req.memory_id)

        queue = MemoryIngestionQueue(handler)
        queue.start()
        await queue.enqueue(self._make_request("test-1"))
        await queue.shutdown(drain=True)

        assert "test-1" in processed

    @pytest.mark.asyncio
    async def test_enqueue_after_shutdown_raises(self):
        """Enqueue after shutdown should raise RuntimeError."""

        async def handler(req: IngestionRequest) -> None:
            pass

        queue = MemoryIngestionQueue(handler)
        queue.start()
        await queue.shutdown(drain=True)

        with pytest.raises(RuntimeError, match="shutting down"):
            await queue.enqueue(self._make_request("test-2"))

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_crash_worker(self):
        """Worker should survive handler exceptions."""
        call_count = 0

        async def handler(req: IngestionRequest) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first call fails")

        queue = MemoryIngestionQueue(handler)
        queue.start()

        for i in range(3):
            await queue.enqueue(self._make_request(f"test-{i}"))

        await queue.shutdown(drain=True)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Calling start() multiple times should not create multiple workers."""

        async def handler(req: IngestionRequest) -> None:
            pass

        queue = MemoryIngestionQueue(handler)
        queue.start()
        first_task = queue._worker_task
        queue.start()
        assert queue._worker_task is first_task
        await queue.shutdown(drain=True)
