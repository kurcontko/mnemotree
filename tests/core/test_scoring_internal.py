"""Tests for core._internal.scoring — keyword overlap and signal ranking."""

from __future__ import annotations

import pytest

from mnemotree.core._internal.scoring import (
    SignalRanker,
    keyword_overlap_score,
)
from mnemotree.core.models import MemoryItem, MemoryType


def _mem(
    mid: str, *, tags: list[str] | None = None, embedding: list[float] | None = None
) -> MemoryItem:
    return MemoryItem(
        memory_id=mid,
        content=f"content-{mid}",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        tags=tags or [],
        embedding=embedding,
    )


class TestKeywordOverlapScore:
    def test_exact_match(self):
        assert keyword_overlap_score(["python", "testing"], ["python"]) == pytest.approx(1.0)

    def test_partial_match(self):
        assert keyword_overlap_score(["python", "testing"], ["python", "rust"]) == pytest.approx(
            0.5
        )

    def test_no_match(self):
        assert keyword_overlap_score(["python"], ["rust"]) == pytest.approx(0.0)

    def test_none_tags(self):
        assert keyword_overlap_score(None, ["python"]) == pytest.approx(0.0)

    def test_none_keywords(self):
        assert keyword_overlap_score(["python"], None) == pytest.approx(0.0)

    def test_empty_tags(self):
        assert keyword_overlap_score([], ["python"]) == pytest.approx(0.0)

    def test_empty_keywords(self):
        assert keyword_overlap_score(["python"], []) == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert keyword_overlap_score(["Python"], ["python"]) == pytest.approx(1.0)

    def test_hyphenated_tags_split(self):
        # "machine-learning" should split into "machine", "learning"
        score = keyword_overlap_score(["machine-learning"], ["machine"])
        assert score == pytest.approx(1.0)


class TestSignalRanker:
    def test_empty_memories(self):
        ranker = SignalRanker()
        assert ranker.rank([], None, None) == []

    def test_rank_by_entity_boost(self):
        m1 = _mem("m1", embedding=[1.0, 0.0])
        m2 = _mem("m2", embedding=[1.0, 0.0])
        ranker = SignalRanker(entity_boost=0.5)
        result = ranker.rank(
            [m1, m2],
            query_embedding=[1.0, 0.0],
            extra_signals={"entity_memory_ids": {"m2"}},
        )
        assert result[0].memory_id == "m2"

    def test_rank_by_rrf(self):
        m1 = _mem("m1", embedding=[1.0, 0.0])
        m2 = _mem("m2", embedding=[1.0, 0.0])
        ranker = SignalRanker(rrf_weight=1.0, entity_boost=0.0, keyword_boost=0.0)
        result = ranker.rank(
            [m1, m2],
            query_embedding=[1.0, 0.0],
            extra_signals={"rrf_scores": {"m2": 10.0, "m1": 1.0}},
        )
        assert result[0].memory_id == "m2"

    def test_none_embedding_graceful(self):
        m1 = _mem("m1", embedding=None)
        ranker = SignalRanker()
        result = ranker.rank([m1], query_embedding=None, extra_signals=None)
        assert len(result) == 1
