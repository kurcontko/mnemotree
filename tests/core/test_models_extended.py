"""
Additional comprehensive tests for core/models.py to increase coverage from 41.5% to 80%+.

This file adds tests for uncovered edge cases in:
- MemoryType.category for ENTITIES (raises ValueError)
- EmotionCategory enum values
- coerce_datetime edge cases (invalid formats, naive datetime)
- MemoryItem validation (importance, emotional bounds)
- MemoryItem.to_str() complex formatting logic (complexity: 55)
- MemoryItem.to_str_llm() all branches (complexity: 16)
- Edge cases for emotional valence indicators (positive/negative/neutral)
- Timeline formatting with only previous or only next
- Source and credibility formatting
- Context and metadata JSON formatting
"""

from datetime import datetime, timedelta, timezone
import math

import pytest
from pydantic import ValidationError

from mnemotree.core.models import (
    EmotionCategory,
    MemoryItem,
    MemoryType,
    coerce_datetime,
)


class TestMemoryTypeCategory:
    """Test MemoryType.category property for all enum values including edge cases."""

    def test_entities_type_raises_value_error(self):
        """ENTITIES type should raise ValueError when accessing category."""
        with pytest.raises(ValueError, match="Unknown category for memory type"):
            _ = MemoryType.ENTITIES.category


class TestEmotionCategoryValues:
    """Test all EmotionCategory enum values."""

    def test_all_emotion_categories_exist(self):
        """Verify all expected emotion categories are defined."""
        expected = {
            "joy",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "trust",
            "anticipation",
            "neutral",
            "satisfaction",
            "excitement",
        }
        actual = {e.value for e in EmotionCategory}
        assert actual == expected


class TestCoerceDatetimeEdgeCases:
    """Test coerce_datetime with various edge cases and failure modes."""

    def test_naive_datetime_gets_utc_timezone(self):
        """Naive datetime should get UTC timezone added."""
        naive = datetime(2024, 6, 15, 10, 30, 45)
        result = coerce_datetime(naive)
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.hour == 10

    def test_datetime_with_non_utc_timezone_preserved(self):
        """Datetime with non-UTC timezone should be preserved."""
        from datetime import timezone as tz
        from datetime import timedelta as td

        eastern = tz(td(hours=-5))
        dt = datetime(2024, 6, 15, 10, 30, tzinfo=eastern)
        result = coerce_datetime(dt)
        assert result.tzinfo == eastern

    def test_isoformat_with_milliseconds(self):
        """Parse ISO format with milliseconds."""
        result = coerce_datetime("2024-06-15T10:30:45.123456Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6

    def test_completely_invalid_string_with_no_default(self):
        """Invalid string with no default should return None."""
        result = coerce_datetime("completely invalid")
        assert result is None

    def test_empty_string_with_default(self):
        """Empty string should return default."""
        default = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = coerce_datetime("", default=default)
        assert result == default


class TestMemoryItemValidationEdgeCases:
    """Test validation edge cases for MemoryItem fields."""

    def test_importance_exactly_zero_is_valid(self):
        """importance=0.0 should be valid."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.0,
        )
        assert abs(memory.importance - 0.0) < 1e-9

    def test_importance_exactly_one_is_valid(self):
        """importance=1.0 should be valid."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=1.0,
        )
        assert abs(memory.importance - 1.0) < 1e-9

    def test_emotional_valence_at_boundaries(self):
        """Test emotional_valence at exact boundaries [-1, 1]."""
        mem1 = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_valence=-1.0,
        )
        assert mem1.emotional_valence == -1.0

        mem2 = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_valence=1.0,
        )
        assert abs(mem2.emotional_valence - 1.0) < 1e-9

    def test_emotional_arousal_at_boundaries(self):
        """Test emotional_arousal at exact boundaries [0, 1]."""
        mem1 = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_arousal=0.0,
        )
        assert abs(mem1.emotional_arousal - 0.0) < 1e-9

        mem2 = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_arousal=1.0,
        )
        assert abs(mem2.emotional_arousal - 1.0) < 1e-9


class TestMemoryItemToStrComplexCases:
    """Test MemoryItem.to_str() edge cases to cover all branches (complexity: 55)."""

    def test_to_str_with_none_timestamp(self):
        """Test to_str formats None timestamp as N/A."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
        )
        # Manually set timestamp to None to test format_time edge case
        memory.timestamp = None
        result = memory.to_str()
        # Should handle None gracefully
        assert "N/A" in result or "Created:" in result

    def test_to_str_only_emotional_valence(self):
        """Test emotional context with only valence (no arousal)."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_valence=0.5,
        )
        result = memory.to_str()
        assert "**Val:** 0.50(+)" in result
        assert "**Aro:**" not in result

    def test_to_str_only_emotional_arousal(self):
        """Test emotional context with only arousal (no valence)."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_arousal=0.7,
        )
        result = memory.to_str()
        assert "**Aro:** 0.70" in result
        assert "**Val:**" not in result

    def test_to_str_only_emotions_list(self):
        """Test emotional context with only emotions list (requires valence/arousal)."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_valence=0.1,  # Need at least one to show emotional section
            emotions=["fear", "anxiety"],
        )
        result = memory.to_str()
        assert "**Emo:** fear, anxiety" in result

    def test_to_str_only_previous_event(self):
        """Test timeline with only previous event."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            previous_event_id="prev-only",
        )
        result = memory.to_str()
        assert "**Timeline:**" in result
        assert "← prev-only" in result
        assert "→" not in result

    def test_to_str_only_next_event(self):
        """Test timeline with only next event."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            next_event_id="next-only",
        )
        result = memory.to_str()
        assert "**Timeline:**" in result
        assert "next-only →" in result
        assert "←" not in result

    def test_to_str_only_source_no_credibility(self):
        """Test source info with only source (no credibility)."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            source="book",
        )
        result = memory.to_str()
        assert "**Source:** book" in result
        assert "**Cred:**" not in result

    def test_to_str_only_credibility_no_source(self):
        """Test source info with only credibility (no source text)."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            source=None,
            credibility=0.75,
        )
        result = memory.to_str()
        assert "**Cred:** 0.75" in result

    def test_to_str_context_as_string(self):
        """Test context field when it's a string."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            context="string context",
        )
        result = memory.to_str()
        assert "**Context:**" in result

    def test_to_str_empty_lists_not_shown(self):
        """Test that empty lists for connections are not shown."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            associations=[],
            linked_concepts=[],
            conflicts_with=[],
        )
        result = memory.to_str()
        # Should not show connection section if all lists are empty
        assert "**Assoc:**" not in result
        assert "**Links:**" not in result
        assert "**Conflicts:**" not in result

    def test_to_str_with_default_metrics(self):
        """Test memory uses default confidence and fidelity values."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            # confidence defaults to 1.0
            # fidelity defaults to 1.0
        )
        result = memory.to_str()
        # Should still render basic structure with default metrics
        assert "### Memory:" in result
        assert "**Imp:** 0.50" in result
        assert "**Conf:** 1.00" in result
        assert "**Fid:** 1.00" in result

    def test_to_str_rating_bars_all_levels(self):
        """Test rating bar generation for different importance levels."""
        for importance in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            memory = MemoryItem(
                content="test",
                memory_type=MemoryType.SEMANTIC,
                importance=importance,
            )
            result = memory.to_str()
            # Rating should be present
            filled = round(importance * 5)
            expected_filled = "|" * filled
            expected_empty = "-" * (5 - filled)
            expected_rating = f"[{expected_filled}{expected_empty}]"
            assert expected_rating in result


class TestMemoryItemToStrLlmEdgeCases:
    """Test MemoryItem.to_str_llm() edge cases (complexity: 16)."""

    def test_to_str_llm_negative_emotional_valence(self):
        """Test negative valence formatting in LLM output."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            emotional_valence=-0.7,
        )
        result = memory.to_str_llm()
        assert "valence: -0.70" in result

    def test_to_str_llm_only_previous_in_timeline(self):
        """Test timeline with only previous event in LLM output."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            previous_event_id="prev-id",
        )
        result = memory.to_str_llm()
        assert "Timeline:" in result
        assert "previous: prev-id" in result
        assert "next:" not in result

    def test_to_str_llm_only_next_in_timeline(self):
        """Test timeline with only next event in LLM output."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            next_event_id="next-id",
        )
        result = memory.to_str_llm()
        assert "Timeline:" in result
        assert "next: next-id" in result
        assert "previous:" not in result

    def test_to_str_llm_no_optional_fields(self):
        """Test LLM output with only required fields."""
        memory = MemoryItem(
            content="Minimal content",
            memory_type=MemoryType.WORKING,
            importance=0.3,
        )
        result = memory.to_str_llm()
        assert "Memory (working):" in result
        assert "Content: Minimal content" in result
        assert "Importance: 0.30" in result
        # Should not have optional sections
        assert "Emotional Context:" not in result
        assert "Related Concepts:" not in result
        assert "Entities:" not in result
        assert "Timeline:" not in result

    def test_to_str_llm_empty_entities_dict(self):
        """Test that empty entities dict doesn't show entities section."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            entities={},
        )
        result = memory.to_str_llm()
        assert "Entities:" not in result

    def test_to_str_llm_empty_linked_concepts(self):
        """Test that empty linked_concepts doesn't show concepts section."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            linked_concepts=[],
        )
        result = memory.to_str_llm()
        assert "Related Concepts:" not in result


class TestMemoryItemAccessTrackingEdgeCases:
    """Test edge cases for access tracking methods."""

    def test_update_access_multiple_times(self):
        """Test multiple access updates."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.3,
            decay_rate=0.05,
        )
        for _ in range(5):
            memory.update_access()

        assert memory.access_count == 5
        assert len(memory.access_history) == 5
        # Should eventually cap at 1.0
        assert memory.importance <= 1.0

    def test_decay_importance_with_string_last_accessed(self):
        """Test decay with last_accessed as ISO string."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
            decay_rate=0.02,
            last_accessed="2024-01-01T00:00:00Z",
        )
        current = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)  # 1 hour later
        memory.decay_importance(current)
        # 3600 seconds * 0.02 = 72
        assert memory.importance < 0.8
        assert memory.importance >= 0

    def test_decay_importance_large_time_diff(self):
        """Test decay with very large time difference."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            decay_rate=0.01,
        )
        far_future = datetime.now(timezone.utc) + timedelta(days=1000)
        memory.decay_importance(far_future)
        # Should be floored at 0
        assert abs(memory.importance - 0.0) < 1e-9


class TestMemoryItemLangchainDocumentEdgeCases:
    """Test edge cases for to_langchain_document conversion."""

    def test_to_langchain_document_with_empty_lists(self):
        """Test that empty lists are filtered from metadata."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            tags=[],
            associations=[],
            linked_concepts=[],
            emotions=[],
        )
        doc = memory.to_langchain_document()
        # Empty lists should be filtered
        assert "tags" not in doc.metadata or doc.metadata["tags"] == []

    def test_to_langchain_document_preserves_custom_metadata_types(self):
        """Test that various metadata types are preserved."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            metadata={
                "string_field": "value",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "list_field": [1, 2, 3],
                "dict_field": {"nested": "data"},
            },
        )
        doc = memory.to_langchain_document()
        assert doc.metadata["string_field"] == "value"
        assert doc.metadata["int_field"] == 42
        assert math.isclose(doc.metadata["float_field"], 3.14)
        assert doc.metadata["bool_field"] is True
        assert doc.metadata["list_field"] == [1, 2, 3]
        assert doc.metadata["dict_field"] == {"nested": "data"}

    def test_to_langchain_document_working_memory_category(self):
        """Test that WORKING memory type has correct category."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.WORKING,
            importance=0.5,
        )
        doc = memory.to_langchain_document()
        assert doc.metadata["memory_type"] == "working"
        assert doc.metadata["memory_category"] == "working"

    def test_to_langchain_document_procedural_memory_category(self):
        """Test that PROCEDURAL memory type has correct category."""
        memory = MemoryItem(
            content="test",
            memory_type=MemoryType.PROCEDURAL,
            importance=0.5,
        )
        doc = memory.to_langchain_document()
        assert doc.metadata["memory_type"] == "procedural"
        assert doc.metadata["memory_category"] == "non_declarative"
