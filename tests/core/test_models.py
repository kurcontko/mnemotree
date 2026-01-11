from __future__ import annotations

from datetime import datetime, timezone

from mnemotree.core.models import MemoryItem, MemoryType, coerce_datetime


def _memory(**overrides) -> MemoryItem:
    data = {
        "memory_id": "mem-1",
        "content": "Hello world",
        "memory_type": MemoryType.SEMANTIC,
        "importance": 0.5,
    }
    data.update(overrides)
    return MemoryItem(**data)


def test_coerce_datetime_parses_iso_and_normalizes_timezone():
    iso = "2024-01-01T12:34:56Z"
    parsed = coerce_datetime(iso)

    assert parsed == datetime(2024, 1, 1, 12, 34, 56, tzinfo=timezone.utc)

    naive = datetime(2024, 2, 1, 9, 0, 0)
    normalized = coerce_datetime(naive)

    assert normalized.tzinfo == timezone.utc


def test_coerce_datetime_falls_back_to_default():
    fallback = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert coerce_datetime("not-a-date", default=fallback) == fallback
    assert coerce_datetime(None, default=None) is None


def test_update_access_tracks_history_and_caps_importance():
    memory = _memory(importance=0.95, decay_rate=0.1)

    memory.update_access()

    assert memory.access_count == 1
    assert memory.last_accessed == memory.access_history[-1]
    assert abs(memory.importance - 1.0) < 1e-9


def test_decay_importance_uses_last_accessed_value():
    last_accessed = "2024-01-01 00:00:00.000000+0000"
    current_time = datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    memory = _memory(importance=1.0, decay_rate=0.1, last_accessed=last_accessed)

    memory.decay_importance(current_time)

    assert abs(memory.importance - 0.5) < 1e-9


def test_to_str_includes_key_sections():
    memory = _memory(
        memory_id="mem-42",
        content="Remember this",
        summary="Short summary",
        tags=["alpha", "beta"],
        memory_type=MemoryType.EPISODIC,
        importance=0.6,
        confidence=0.7,
        fidelity=0.8,
        emotional_valence=0.2,
        emotional_arousal=0.4,
        emotions=["joy", "trust"],
        associations=["mem-2"],
        linked_concepts=["concept-1"],
        conflicts_with=["mem-3"],
        previous_event_id="mem-1",
        next_event_id="mem-99",
        source="imported",
        credibility=0.9,
        context={"place": "home"},
        metadata={"topic": "test"},
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    rendered = memory.to_str()

    assert "### Memory: mem-42" in rendered
    assert "Short summary" in rendered
    assert "**Tags:** alpha, beta" in rendered
    assert "Imp:" in rendered
    assert "**Emo:** joy, trust" in rendered
    assert "**Assoc:** mem-2" in rendered
    assert "**Links:** concept-1" in rendered
    assert "**Conflicts:** mem-3" in rendered
    assert "Timeline:" in rendered
    assert "**Source:** imported" in rendered
    assert "Context:" in rendered
    assert "Metadata:" in rendered


def test_to_str_llm_highlights_entities_emotions_and_timeline():
    memory = _memory(
        content="Coffee with Alice",
        summary="Met Alice",
        importance=0.42,
        emotional_valence=0.3,
        emotional_arousal=0.6,
        emotions=["joy"],
        linked_concepts=["coffee"],
        entities={"Alice": "PERSON"},
        previous_event_id="mem-1",
        next_event_id="mem-3",
    )

    rendered = memory.to_str_llm()

    assert "Memory (semantic):" in rendered
    assert "Content: Coffee with Alice" in rendered
    assert "Summary: Met Alice" in rendered
    assert "Emotional Context:" in rendered
    assert "Related Concepts: coffee" in rendered
    assert "Entities: Alice (PERSON)" in rendered
    assert "Timeline: previous: mem-1 | next: mem-3" in rendered


def test_to_langchain_document_includes_metadata():
    memory = _memory(
        memory_id="mem-5",
        content="LangChain test",
        importance=0.9,
        tags=["tag-1"],
        emotional_valence=0.1,
        emotional_arousal=0.2,
        emotions=["joy"],
        metadata={"source_id": "abc"},
    )

    doc = memory.to_langchain_document()

    assert doc.page_content == "LangChain test"
    assert doc.metadata["memory_id"] == "mem-5"
    assert doc.metadata["memory_type"] == MemoryType.SEMANTIC.value
    assert doc.metadata["memory_category"] == MemoryType.SEMANTIC.category
    assert doc.metadata["tags"] == ["tag-1"]
    assert doc.metadata["source_id"] == "abc"
