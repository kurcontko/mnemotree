"""Unit tests for gap detection (src/mnemotree/core/gap_detection.py)."""
from __future__ import annotations

import pytest

from mnemotree.core.gap_detection import HeuristicGapAnalyzer, _extract_wh_word
from mnemotree.core.models import MemoryItem, MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory(content: str, entities: dict[str, str] | None = None) -> MemoryItem:
    return MemoryItem(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        entities=entities or {},
    )


# ---------------------------------------------------------------------------
# _extract_wh_word
# ---------------------------------------------------------------------------


def test_extract_wh_word_who():
    assert _extract_wh_word("Who invented the transistor?") == "who"


def test_extract_wh_word_where():
    assert _extract_wh_word("Where does Alice work?") == "where"


def test_extract_wh_word_when():
    assert _extract_wh_word("When was Python created?") == "when"


def test_extract_wh_word_none():
    assert _extract_wh_word("Tell me about Bob.") is None


def test_extract_wh_word_case_insensitive():
    assert _extract_wh_word("WHO runs ACME corp?") == "who"


# ---------------------------------------------------------------------------
# HeuristicGapAnalyzer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heuristic_no_gaps_when_entities_covered():
    analyzer = HeuristicGapAnalyzer()
    retrieved = [_memory("Alice works at ACME.", entities={"Alice": "PERSON", "ACME": "ORG"})]
    sub_queries = await analyzer.detect_gaps(
        "What does Alice do at ACME?",
        retrieved,
        {"Alice": "PERSON", "ACME": "ORG"},
    )
    assert sub_queries == []


@pytest.mark.asyncio
async def test_heuristic_gap_for_missing_entity():
    analyzer = HeuristicGapAnalyzer()
    retrieved = [_memory("Bob works at ACME.", entities={"ACME": "ORG"})]
    # Alice is in query but not in retrieved memories
    sub_queries = await analyzer.detect_gaps(
        "What does Alice do?",
        retrieved,
        {"Alice": "PERSON"},
    )
    assert any("Alice" in sq for sq in sub_queries)


@pytest.mark.asyncio
async def test_heuristic_respects_max_gaps():
    analyzer = HeuristicGapAnalyzer(max_gaps=1)
    retrieved = [_memory("Some content.")]
    sub_queries = await analyzer.detect_gaps(
        "Who is Alice, Bob, and Charlie?",
        retrieved,
        {"Alice": "PERSON", "Bob": "PERSON", "Charlie": "PERSON"},
    )
    assert len(sub_queries) <= 1


@pytest.mark.asyncio
async def test_heuristic_no_retrieved_memories():
    analyzer = HeuristicGapAnalyzer()
    sub_queries = await analyzer.detect_gaps(
        "Where does Bob work?",
        [],
        {"Bob": "PERSON"},
    )
    assert isinstance(sub_queries, list)
    # Should generate a gap query for Bob
    assert len(sub_queries) >= 1


@pytest.mark.asyncio
async def test_heuristic_wh_question_gap():
    """A 'who' question should trigger a wh-word gap if no PERSON found."""
    analyzer = HeuristicGapAnalyzer()
    # Retrieved memories mention a location but no person
    retrieved = [_memory("The lab is in Cambridge.", entities={"Cambridge": "LOC"})]
    sub_queries = await analyzer.detect_gaps(
        "Who runs the lab in Cambridge?",
        retrieved,
        {},
    )
    # Should detect that a PERSON-type answer is missing
    assert len(sub_queries) >= 1
