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
) -> MemoryItem:
    return MemoryItem(
        content="Test content",
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        timestamp=_ts(timestamp),
        access_count=access_count,
        embedding=embedding,
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
    assert score_opposite == 0.0


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
