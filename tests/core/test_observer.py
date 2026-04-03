"""Tests for Observer/Reflector pattern (Mastra OM)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.observer import (
    MemoryObserver,
    MemoryReflector,
    Observation,
    ObservationConfig,
    ReflectorConfig,
    ReflectorResult,
)

# ---------------------------------------------------------------------------
# Observer tests
# ---------------------------------------------------------------------------


class TestObservationConfig:
    def test_defaults(self):
        config = ObservationConfig()
        assert config.min_content_length == 15
        assert config.max_observations_per_turn == 5
        assert config.default_memory_type == MemoryType.EPISODIC
        assert config.use_llm_extraction is True

    def test_custom(self):
        config = ObservationConfig(
            min_content_length=5,
            max_observations_per_turn=3,
            default_memory_type=MemoryType.SEMANTIC,
            use_llm_extraction=False,
        )
        assert config.min_content_length == 5
        assert config.default_memory_type == MemoryType.SEMANTIC


class TestMemoryObserverPassthrough:
    """Test observer in passthrough mode (no LLM)."""

    async def test_passthrough_stores_content(self):
        observer = MemoryObserver(config=ObservationConfig(use_llm_extraction=False))
        result = await observer.observe("The user prefers dark mode for coding")
        assert len(result) == 1
        assert result[0].content == "The user prefers dark mode for coding"
        assert result[0].source == "observer"
        assert result[0].observation_date is not None

    async def test_skips_short_content(self):
        observer = MemoryObserver(config=ObservationConfig(use_llm_extraction=False))
        result = await observer.observe("hi")
        assert len(result) == 0

    async def test_passthrough_sets_observation_date(self):
        observer = MemoryObserver(config=ObservationConfig(use_llm_extraction=False))
        before = datetime.now(timezone.utc)
        result = await observer.observe("User mentioned they work at Acme Corp")
        after = datetime.now(timezone.utc)
        assert len(result) == 1
        assert before <= result[0].observation_date <= after

    async def test_respects_max_observations(self):
        config = ObservationConfig(
            use_llm_extraction=False,
            max_observations_per_turn=1,
        )
        observer = MemoryObserver(config=config)
        result = await observer.observe("User prefers Python over JavaScript")
        assert len(result) <= 1


class TestMemoryObserverLLM:
    """Test observer with LLM extraction."""

    async def test_llm_extracts_facts(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="User prefers dark mode\nUser works at Acme Corp"
        )
        observer = MemoryObserver(llm=mock_llm)
        result = await observer.observe("I really like dark mode. I work at Acme Corp.")
        assert len(result) == 2
        assert result[0].content == "User prefers dark mode"
        assert result[1].content == "User works at Acme Corp"

    async def test_llm_returns_none(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="NONE")
        observer = MemoryObserver(llm=mock_llm)
        result = await observer.observe("Hello, how are you doing today?")
        assert len(result) == 0

    async def test_llm_strips_numbering(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="1. User prefers dark mode\n2. User works at Acme Corp"
        )
        observer = MemoryObserver(llm=mock_llm)
        result = await observer.observe("I use dark mode and work at Acme Corp.")
        assert result[0].content == "User prefers dark mode"
        assert result[1].content == "User works at Acme Corp"

    async def test_llm_failure_falls_back_to_passthrough(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("LLM unavailable")
        observer = MemoryObserver(llm=mock_llm)
        result = await observer.observe("User prefers dark mode for coding")
        assert len(result) == 1
        assert result[0].content == "User prefers dark mode for coding"

    async def test_no_llm_uses_passthrough(self):
        """Even with use_llm_extraction=True, no LLM falls back to passthrough."""
        observer = MemoryObserver(llm=None, config=ObservationConfig(use_llm_extraction=True))
        result = await observer.observe("User prefers dark mode for coding")
        assert len(result) == 1

    async def test_caps_observations_per_turn(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="\n".join(f"Fact number {i} about the user" for i in range(20))
        )
        config = ObservationConfig(max_observations_per_turn=3)
        observer = MemoryObserver(llm=mock_llm, config=config)
        result = await observer.observe("A very long conversation with many facts.")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Reflector tests
# ---------------------------------------------------------------------------


class TestReflectorConfig:
    def test_defaults(self):
        config = ReflectorConfig()
        assert config.min_memories_for_consolidation == 10
        assert config.consolidation_interval_hours == 24.0
        assert config.conflict_similarity_threshold == 0.92


def _make_memory(content: str, embedding: list[float] | None = None) -> MemoryItem:
    """Helper to create a MemoryItem with required fields."""
    return MemoryItem(
        content=content,
        memory_type=MemoryType.EPISODIC,
        importance=0.5,
        embedding=embedding or [0.1, 0.2],
    )


class TestMemoryReflector:
    async def test_should_run_first_time(self):
        reflector = MemoryReflector()
        assert await reflector.should_run(memory_count=20) is True

    async def test_should_not_run_too_few_memories(self):
        reflector = MemoryReflector(config=ReflectorConfig(min_memories_for_consolidation=10))
        assert await reflector.should_run(memory_count=5) is False

    async def test_should_not_run_too_soon(self):
        reflector = MemoryReflector(config=ReflectorConfig(consolidation_interval_hours=24.0))
        reflector._last_run = datetime.now(timezone.utc)
        assert await reflector.should_run(memory_count=20) is False

    async def test_run_with_conflict_judge(self):
        memories = [
            _make_memory("Python is fast"),
            _make_memory("Python is slow"),
        ]
        mock_judge = MagicMock()
        mock_judge.detect_conflicts.return_value = {
            memories[0].memory_id: {"conflicting_ids": [memories[1].memory_id]}
        }

        reflector = MemoryReflector()
        result = await reflector.run(memories, conflict_judge=mock_judge)

        assert isinstance(result, ReflectorResult)
        assert result.memories_processed == 2
        assert result.conflicts_found == 1
        assert result.run_at is not None

    async def test_run_with_consolidator(self):
        memories = [
            _make_memory("User likes Python"),
            _make_memory("User likes coding"),
        ]
        mock_consolidator = AsyncMock()
        mock_consolidator.consolidate.return_value = MagicMock(
            clusters_formed=1, semantic_memories_created=1
        )

        reflector = MemoryReflector()
        result = await reflector.run(memories, consolidator=mock_consolidator)

        assert result.memories_processed == 2
        assert result.consolidation_result is not None
        mock_consolidator.consolidate.assert_called_once_with(memories)

    async def test_run_empty_memories(self):
        reflector = MemoryReflector()
        result = await reflector.run([])
        assert result.memories_processed == 0
        assert result.conflicts_found == 0

    async def test_run_updates_last_run(self):
        reflector = MemoryReflector()
        assert reflector._last_run is None
        await reflector.run([])
        assert reflector._last_run is not None

    async def test_run_handles_conflict_judge_failure(self):
        memories = [_make_memory("Some content here")]
        mock_judge = MagicMock()
        mock_judge.detect_conflicts.side_effect = RuntimeError("judge failed")

        reflector = MemoryReflector()
        result = await reflector.run(memories, conflict_judge=mock_judge)
        assert result.conflicts_found == 0

    async def test_run_handles_consolidator_failure(self):
        memories = [_make_memory("Some content here")]
        mock_consolidator = AsyncMock()
        mock_consolidator.consolidate.side_effect = RuntimeError("consolidation failed")

        reflector = MemoryReflector()
        result = await reflector.run(memories, consolidator=mock_consolidator)
        assert result.consolidation_result is None


# ---------------------------------------------------------------------------
# Observation dataclass tests
# ---------------------------------------------------------------------------


class TestObservation:
    def test_defaults(self):
        obs = Observation(content="test fact")
        assert obs.memory_type == MemoryType.EPISODIC
        assert obs.importance == 0.5
        assert obs.source == "observer"
        assert obs.tags == []

    def test_custom(self):
        now = datetime.now(timezone.utc)
        obs = Observation(
            content="User prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            observation_date=now,
            tags=["preference"],
        )
        assert obs.importance == 0.8
        assert obs.observation_date == now
        assert obs.tags == ["preference"]
