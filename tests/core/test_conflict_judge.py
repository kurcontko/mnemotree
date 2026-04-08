"""Tests for the recall-time conflict judge (AMA Judge pattern)."""

from mnemotree.core._internal.conflict_judge import (
    ConflictAnnotation,
    ConflictJudge,
    ConflictJudgeConfig,
)
from mnemotree.core.models import MemoryItem, MemoryType


def _memory(
    memory_id: str,
    content: str,
    embedding: list[float] | None = None,
    conflicts_with: list[str] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=embedding,
        conflicts_with=conflicts_with or [],
    )


class TestConflictJudge:
    def test_no_conflicts_when_dissimilar(self) -> None:
        judge = ConflictJudge()
        m1 = _memory("a", "Python is great", [1.0, 0.0, 0.0])
        m2 = _memory("b", "The sky is blue", [0.0, 1.0, 0.0])
        result = judge.detect_conflicts([m1, m2])
        assert result == {}

    def test_detects_preexisting_conflicts(self) -> None:
        judge = ConflictJudge()
        m1 = _memory("a", "Python is fast", [1.0, 0.0], conflicts_with=["b"])
        m2 = _memory("b", "Python is slow", [0.9, 0.1])
        result = judge.detect_conflicts([m1, m2])
        assert "a" in result
        assert "b" in result
        assert "b" in result["a"].conflicting_ids
        assert "a" in result["b"].conflicting_ids

    def test_detects_negation_conflict(self) -> None:
        # Same topic with negation → high sim + negation signal
        judge = ConflictJudge(config=ConflictJudgeConfig(similarity_threshold=0.9))
        m1 = _memory("a", "Python supports threading", [1.0, 0.0])
        m2 = _memory("b", "Python does not support threading", [0.99, 0.01])
        result = judge.detect_conflicts([m1, m2])
        assert "a" in result
        assert "b" in result["a"].conflicting_ids
        assert "negation" in result["a"].reasons["b"]

    def test_detects_near_duplicate(self) -> None:
        judge = ConflictJudge(config=ConflictJudgeConfig(similarity_threshold=0.95))
        from datetime import datetime, timezone

        m1 = _memory("old", "Python is great for scripting", [1.0, 0.0])
        m1.timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        m2 = _memory("new", "Python is great for scripting", [1.0, 0.0])
        m2.timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = judge.detect_conflicts([m1, m2])
        # Near-duplicate detected (newer may supersede older)
        assert len(result) >= 1

    def test_disabled_returns_empty(self) -> None:
        judge = ConflictJudge(config=ConflictJudgeConfig(enabled=False))
        m1 = _memory("a", "X", [1.0], conflicts_with=["b"])
        m2 = _memory("b", "Y", [1.0])
        assert judge.detect_conflicts([m1, m2]) == {}

    def test_single_memory_returns_empty(self) -> None:
        judge = ConflictJudge()
        m1 = _memory("a", "X", [1.0])
        assert judge.detect_conflicts([m1]) == {}

    def test_max_pairs_limit(self) -> None:
        judge = ConflictJudge(config=ConflictJudgeConfig(max_pairs_checked=1))
        memories = [_memory(f"m{i}", f"content {i}", [float(i)]) for i in range(10)]
        # Should not error — just stops checking early
        result = judge.detect_conflicts(memories)
        assert isinstance(result, dict)

    def test_no_embeddings_skips_cosine(self) -> None:
        judge = ConflictJudge()
        m1 = _memory("a", "X", embedding=None)
        m2 = _memory("b", "Y", embedding=None)
        result = judge.detect_conflicts([m1, m2])
        assert result == {}


class TestConflictAnnotation:
    def test_default_fields(self) -> None:
        ann = ConflictAnnotation(memory_id="test")
        assert ann.memory_id == "test"
        assert ann.conflicting_ids == []
        assert ann.reasons == {}
