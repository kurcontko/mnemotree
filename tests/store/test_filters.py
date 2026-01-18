"""Tests for store filter clause builders."""

from datetime import datetime, timezone

import pytest

from mnemotree.core.models import MemoryType
from mnemotree.core.query import FilterOperator, MemoryFilter
from mnemotree.store._filters import build_sqlite_filter_clauses, normalize_filter_value
from mnemotree.store.query_builders import UnsupportedQueryError


class TestNormalizeFilterValue:
    """Tests for normalize_filter_value in _filters module."""

    def test_memory_type_to_value(self):
        """MemoryType enum becomes its string value."""
        assert normalize_filter_value(MemoryType.SEMANTIC) == "semantic"
        assert normalize_filter_value(MemoryType.EPISODIC) == "episodic"

    def test_datetime_to_isoformat(self):
        """datetime becomes ISO format string."""
        dt = datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)
        assert normalize_filter_value(dt) == "2024-06-15T12:30:00+00:00"

    def test_string_passthrough(self):
        """Strings pass through unchanged."""
        assert normalize_filter_value("hello") == "hello"

    def test_int_passthrough(self):
        """Integers pass through unchanged."""
        assert normalize_filter_value(42) == 42


class TestBuildSqliteFilterClauses:
    """Tests for build_sqlite_filter_clauses()."""

    # --- EQ and NE operators ---

    def test_eq_operator(self):
        """EQ operator generates correct clause."""
        filters = [MemoryFilter(field="memory_type", operator=FilterOperator.EQ, value="semantic")]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["memory_type = ?"]
        assert params == ["semantic"]

    def test_eq_operator_memory_id(self):
        """EQ operator supports memory_id."""
        filters = [MemoryFilter(field="memory_id", operator=FilterOperator.EQ, value="mem-123")]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["memory_id = ?"]
        assert params == ["mem-123"]

    def test_eq_operator_event_links(self):
        """EQ operator supports previous/next event fields."""
        filters = [
            MemoryFilter(field="previous_event_id", operator=FilterOperator.EQ, value="prev-1"),
            MemoryFilter(field="next_event_id", operator=FilterOperator.EQ, value="next-1"),
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["previous_event_id = ?", "next_event_id = ?"]
        assert params == ["prev-1", "next-1"]

    def test_ne_operator(self):
        """NE operator generates correct clause."""
        filters = [MemoryFilter(field="source", operator=FilterOperator.NE, value="test")]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["source != ?"]
        assert params == ["test"]

    # --- Comparison operators ---

    def test_gt_operator(self):
        """GT operator generates correct clause."""
        filters = [MemoryFilter(field="importance", operator=FilterOperator.GT, value=0.5)]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["importance > ?"]
        assert params == [0.5]

    def test_gte_operator(self):
        """GTE operator generates correct clause."""
        filters = [MemoryFilter(field="importance", operator=FilterOperator.GTE, value=0.8)]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["importance >= ?"]
        assert params == [0.8]

    def test_lt_operator(self):
        """LT operator generates correct clause."""
        filters = [MemoryFilter(field="confidence", operator=FilterOperator.LT, value=0.9)]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["confidence < ?"]
        assert params == [0.9]

    def test_lte_operator(self):
        """LTE operator generates correct clause."""
        filters = [MemoryFilter(field="confidence", operator=FilterOperator.LTE, value=1.0)]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["confidence <= ?"]
        assert params == [1.0]

    # --- IN and NOT_IN operators ---

    def test_in_operator(self):
        """IN operator generates correct clause."""
        filters = [
            MemoryFilter(
                field="memory_type", operator=FilterOperator.IN, value=["semantic", "episodic"]
            )
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["memory_type IN (?, ?)"]
        assert params == ["semantic", "episodic"]

    def test_not_in_operator(self):
        """NOT_IN operator generates correct clause."""
        filters = [
            MemoryFilter(field="source", operator=FilterOperator.NOT_IN, value=["a", "b", "c"])
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["source NOT IN (?, ?, ?)"]
        assert params == ["a", "b", "c"]

    def test_in_empty_list(self):
        """IN with empty list returns always-false clause."""
        filters = [MemoryFilter(field="memory_type", operator=FilterOperator.IN, value=[])]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["1 = 0"]
        assert params == []

    def test_not_in_empty_list(self):
        """NOT_IN with empty list returns always-true clause."""
        filters = [MemoryFilter(field="memory_type", operator=FilterOperator.NOT_IN, value=[])]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["1 = 1"]
        assert params == []

    # --- CONTAINS operators on content ---

    def test_contains_on_content(self):
        """CONTAINS on content field uses LIKE."""
        filters = [MemoryFilter(field="content", operator=FilterOperator.CONTAINS, value="hello")]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["(content LIKE ?)"]
        assert params == ["%hello%"]

    def test_contains_on_content_multiple_values(self):
        """CONTAINS on content supports multiple values with OR."""
        filters = [
            MemoryFilter(
                field="content",
                operator=FilterOperator.CONTAINS,
                value=["alpha", "beta"],
            )
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["(content LIKE ? OR content LIKE ?)"]
        assert params == ["%alpha%", "%beta%"]

    def test_not_contains_on_content(self):
        """NOT_CONTAINS on content field uses NOT LIKE."""
        filters = [
            MemoryFilter(field="content", operator=FilterOperator.NOT_CONTAINS, value="secret")
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["NOT (content LIKE ?)"]
        assert params == ["%secret%"]

    # --- CONTAINS operators on list fields ---

    def test_contains_on_tags(self):
        """CONTAINS on tags field uses comma-wrapped LIKE."""
        filters = [MemoryFilter(field="tags", operator=FilterOperator.CONTAINS, value="important")]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["((',' || tags || ',') LIKE ?)"]
        assert params == ["%,important,%"]

    def test_contains_on_linked_concepts(self):
        """CONTAINS on linked_concepts supports multiple values with OR."""
        filters = [
            MemoryFilter(
                field="linked_concepts",
                operator=FilterOperator.CONTAINS,
                value=["concept-a", "concept-b"],
            )
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == [
            "((',' || linked_concepts || ',') LIKE ? OR (',' || linked_concepts || ',') LIKE ?)"
        ]
        assert params == ["%,concept-a,%", "%,concept-b,%"]

    def test_not_contains_on_conflicts_with(self):
        """NOT_CONTAINS on conflicts_with uses NOT with comma-wrapped LIKE."""
        filters = [
            MemoryFilter(
                field="conflicts_with",
                operator=FilterOperator.NOT_CONTAINS,
                value="mem-9",
            )
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["NOT ((',' || conflicts_with || ',') LIKE ?)"]
        assert params == ["%,mem-9,%"]

    def test_not_contains_on_emotions(self):
        """NOT_CONTAINS on emotions field uses NOT with comma-wrapped LIKE."""
        filters = [
            MemoryFilter(field="emotions", operator=FilterOperator.NOT_CONTAINS, value="anger")
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["NOT ((',' || emotions || ',') LIKE ?)"]
        assert params == ["%,anger,%"]

    # --- Multiple filters ---

    def test_multiple_filters(self):
        """Multiple filters generate multiple clauses."""
        filters = [
            MemoryFilter(field="memory_type", operator=FilterOperator.EQ, value="semantic"),
            MemoryFilter(field="importance", operator=FilterOperator.GTE, value=0.7),
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert clauses == ["memory_type = ?", "importance >= ?"]
        assert params == ["semantic", 0.7]

    # --- Value normalization ---

    def test_memory_type_enum_normalized(self):
        """MemoryType enum is normalized to string value."""
        filters = [
            MemoryFilter(field="memory_type", operator=FilterOperator.EQ, value=MemoryType.EPISODIC)
        ]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert params == ["episodic"]

    def test_datetime_normalized(self):
        """datetime is normalized to ISO format."""
        dt = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        filters = [MemoryFilter(field="timestamp", operator=FilterOperator.GTE, value=dt)]
        clauses, params = build_sqlite_filter_clauses(filters)
        assert params == ["2024-01-15T10:30:00+00:00"]

    # --- Error cases ---

    def test_unsupported_field_raises(self):
        """Unsupported field raises UnsupportedQueryError."""
        filters = [MemoryFilter(field="unknown_field", operator=FilterOperator.EQ, value="test")]
        with pytest.raises(UnsupportedQueryError, match="Unsupported filter field"):
            build_sqlite_filter_clauses(filters)

    def test_matches_operator_raises(self):
        """MATCHES operator raises UnsupportedQueryError."""
        filters = [MemoryFilter(field="content", operator=FilterOperator.MATCHES, value="pattern")]
        with pytest.raises(UnsupportedQueryError, match="MATCHES.*not supported"):
            build_sqlite_filter_clauses(filters)

    def test_contains_on_non_list_field_raises(self):
        """CONTAINS on non-content, non-list field raises UnsupportedQueryError."""
        filters = [MemoryFilter(field="importance", operator=FilterOperator.CONTAINS, value=0.5)]
        with pytest.raises(UnsupportedQueryError, match="Unsupported filter operator"):
            build_sqlite_filter_clauses(filters)
