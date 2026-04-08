from datetime import datetime, timedelta, timezone

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.scoring import MemoryScoring


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")


def _memory(
    timestamp: datetime,
    *,
    importance: float = 0.5,
    access_count: int = 0,
    embedding: list[float] | None = None,
    last_accessed: datetime | None = None,
    decay_rate: float = 0.01,
) -> MemoryItem:
    return MemoryItem(
        content="Test content",
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        timestamp=_ts(timestamp),
        access_count=access_count,
        embedding=embedding,
        last_accessed=last_accessed or timestamp,
        decay_rate=decay_rate,
    )


def test_recency_monotonic():
    scoring = MemoryScoring(
        importance_weight=0.0,
        recency_weight=1.0,
        relevance_weight=0.0,
        recency_stability_seconds=3600,
        recency_power=-0.5,
    )
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    recent = _memory(now - timedelta(hours=1))
    old = _memory(now - timedelta(hours=10))

    score_recent = scoring.calculate_memory_score(recent, current_time=now)
    score_old = scoring.calculate_memory_score(old, current_time=now)

    assert score_recent > score_old


def test_relevance_positive_cosine():
    scoring = MemoryScoring(
        importance_weight=0.0,
        recency_weight=0.0,
        relevance_weight=1.0,
    )
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)

    memory_match = _memory(now, embedding=[1.0, 0.0])
    score_match = scoring.calculate_memory_score(
        memory_match,
        current_time=now,
        query_embedding=[1.0, 0.0],
    )
    assert score_match == pytest.approx(1.0)

    memory_opposite = _memory(now, embedding=[-1.0, 0.0])
    score_opposite = scoring.calculate_memory_score(
        memory_opposite,
        current_time=now,
        query_embedding=[1.0, 0.0],
    )
    assert score_opposite == pytest.approx(0.0)


def test_importance_access_boost():
    scoring = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
    )
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)

    base = _memory(now, access_count=0, importance=0.5)
    boosted = _memory(now, access_count=100, importance=0.5)

    score_base = scoring.calculate_memory_score(base, current_time=now)
    score_boosted = scoring.calculate_memory_score(boosted, current_time=now)

    assert score_base == pytest.approx(0.5)
    assert score_boosted == pytest.approx(0.7)
    assert score_boosted > score_base


def test_decay_reduces_importance_over_time():
    """Verify importance decreases with time since last access."""
    scoring = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
        enable_decay=True,
        decay_stability_seconds=86400.0,  # 1 day
        decay_floor=0.1,
    )
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)

    recent = _memory(now, importance=0.8, last_accessed=now - timedelta(hours=1))
    old = _memory(now, importance=0.8, last_accessed=now - timedelta(days=7))

    score_recent = scoring.calculate_memory_score(recent, current_time=now)
    score_old = scoring.calculate_memory_score(old, current_time=now)

    assert score_recent > score_old


def test_decay_respects_floor():
    """Verify importance never drops below decay_floor."""
    scoring = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
        enable_decay=True,
        decay_stability_seconds=86400.0,
        decay_floor=0.2,
    )
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)

    # Very old memory should hit the floor
    very_old = _memory(
        now,
        importance=0.5,
        last_accessed=now - timedelta(days=365),
        decay_rate=0.1,
    )

    score = scoring.calculate_memory_score(very_old, current_time=now)
    # Score should be at least decay_floor (0.2)
    assert score >= 0.2


def test_decay_disabled_uses_raw_importance():
    """Verify enable_decay=False uses stored importance without decay."""
    scoring_no_decay = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
        enable_decay=False,
    )
    scoring_with_decay = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
        enable_decay=True,
        decay_stability_seconds=86400.0,
    )
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)

    old = _memory(now, importance=0.8, last_accessed=now - timedelta(days=30))

    score_no_decay = scoring_no_decay.calculate_memory_score(old, current_time=now)
    score_with_decay = scoring_with_decay.calculate_memory_score(old, current_time=now)

    # Without decay, score should equal raw importance
    assert score_no_decay == pytest.approx(0.8)
    # With decay, score should be lower
    assert score_with_decay < score_no_decay


def test_decay_combined_with_access_boost():
    """Verify decay and access boost compose correctly."""
    scoring = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
        enable_decay=True,
        decay_stability_seconds=86400.0,
        decay_floor=0.1,
    )
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)

    # Same memory with/without access count
    base = _memory(now, importance=0.5, last_accessed=now - timedelta(days=3), access_count=0)
    accessed = _memory(now, importance=0.5, last_accessed=now - timedelta(days=3), access_count=100)

    score_base = scoring.calculate_memory_score(base, current_time=now)
    score_accessed = scoring.calculate_memory_score(accessed, current_time=now)

    # Access boost should increase score even with decay
    assert score_accessed > score_base


def test_decay_floor_should_not_increase_importance():
    """Floor should never raise importance above its original value."""
    scoring = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
        enable_decay=True,
        decay_stability_seconds=86400.0,
        decay_floor=0.3,  # Floor is higher than memory's importance
    )
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)

    # Memory with importance BELOW the decay_floor
    low_importance = _memory(
        now,
        importance=0.1,
        last_accessed=now - timedelta(days=1),
        access_count=0,
    )

    score = scoring.calculate_memory_score(low_importance, current_time=now)

    # Score should NOT exceed original importance (0.1)
    # Without the fix, this would incorrectly return 0.3 (the floor)
    assert score <= 0.1


def test_negative_access_count_does_not_crash():
    """Negative access_count (corrupt data) should not cause math.log crash."""
    scoring = MemoryScoring(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
    )
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)

    # Corrupt data: negative access_count
    corrupt = _memory(now, access_count=-5, importance=0.5)
    score = scoring.calculate_memory_score(corrupt, current_time=now)
    # Should not crash; access_boost clamped to log(1) = 0
    assert score == pytest.approx(0.5)
