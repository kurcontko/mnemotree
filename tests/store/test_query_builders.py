from datetime import datetime, timezone

import pytest

from mnemotree.core.models import MemoryType
from mnemotree.core.query import FilterOperator, MemoryFilter
from mnemotree.store.query_builders import (
    UnsupportedQueryError,
    build_chroma_where,
    build_neo4j_where_clause,
)


def test_build_neo4j_where_clause_maps_operators_and_normalizes_enums():
    where, params = build_neo4j_where_clause(
        [
            MemoryFilter("importance", FilterOperator.GTE, 0.7),
            MemoryFilter("memory_type", FilterOperator.EQ, MemoryType.SEMANTIC),
        ]
    )
    assert "m.importance >= $filter_0" in where
    assert "m.memory_type = $filter_1" in where
    assert params == {"filter_0": 0.7, "filter_1": "semantic"}


def test_build_neo4j_where_clause_contains_on_string_uses_contains():
    where, params = build_neo4j_where_clause(
        [MemoryFilter("content", FilterOperator.CONTAINS, "hello")]
    )
    assert where == "WHERE m.content CONTAINS $filter_0"
    assert params == {"filter_0": "hello"}


def test_build_neo4j_where_clause_contains_on_list_uses_in():
    where, params = build_neo4j_where_clause(
        [MemoryFilter("linked_concepts", FilterOperator.CONTAINS, "python")]
    )
    assert where == "WHERE $filter_0 IN m.linked_concepts"
    assert params == {"filter_0": "python"}


def test_build_neo4j_where_clause_normalizes_datetimes():
    ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    where, params = build_neo4j_where_clause([MemoryFilter("timestamp", FilterOperator.GTE, ts)])
    assert where == "WHERE m.timestamp >= $filter_0"
    assert params == {"filter_0": ts.isoformat()}


def test_build_neo4j_where_clause_rejects_relationship_fields():
    with pytest.raises(UnsupportedQueryError):
        build_neo4j_where_clause([MemoryFilter("tags", FilterOperator.CONTAINS, "x")])


def test_build_chroma_where_rejects_non_eq_filters():
    with pytest.raises(UnsupportedQueryError):
        build_chroma_where([MemoryFilter("importance", FilterOperator.GT, 0.5)])
