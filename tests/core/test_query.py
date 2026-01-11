"""Tests for MemoryQuery and MemoryQueryBuilder."""

from datetime import datetime, timezone

import pytest

from mnemotree.core.models import EmotionCategory, MemoryType
from mnemotree.core.query import (
    FilterOperator,
    MemoryFilter,
    MemoryQuery,
    MemoryQueryBuilder,
    MemoryRelationship,
    SortOrder,
)


class TestFilterOperator:
    """Tests for FilterOperator enum."""

    def test_all_operators_exist(self):
        """All expected operators are defined."""
        assert FilterOperator.EQ.value == "eq"
        assert FilterOperator.NE.value == "ne"
        assert FilterOperator.GT.value == "gt"
        assert FilterOperator.GTE.value == "gte"
        assert FilterOperator.LT.value == "lt"
        assert FilterOperator.LTE.value == "lte"
        assert FilterOperator.IN.value == "in"
        assert FilterOperator.NOT_IN.value == "not_in"
        assert FilterOperator.CONTAINS.value == "contains"
        assert FilterOperator.NOT_CONTAINS.value == "not_contains"
        assert FilterOperator.MATCHES.value == "matches"


class TestSortOrder:
    """Tests for SortOrder enum."""

    def test_sort_orders(self):
        """Sort orders are defined correctly."""
        assert SortOrder.ASC.value == "asc"
        assert SortOrder.DESC.value == "desc"


class TestMemoryFilter:
    """Tests for MemoryFilter dataclass."""

    def test_creation(self):
        """MemoryFilter can be created with required fields."""
        filt = MemoryFilter(field="content", operator=FilterOperator.CONTAINS, value="test")
        assert filt.field == "content"
        assert filt.operator == FilterOperator.CONTAINS
        assert filt.value == "test"


class TestMemoryRelationship:
    """Tests for MemoryRelationship dataclass."""

    def test_creation_minimal(self):
        """MemoryRelationship can be created with required fields."""
        rel = MemoryRelationship(type="related_to", direction="out", node_type="Memory")
        assert rel.type == "related_to"
        assert rel.direction == "out"
        assert rel.node_type == "Memory"
        assert rel.condition is None

    def test_creation_with_condition(self):
        """MemoryRelationship can include a condition."""
        condition = MemoryFilter(field="importance", operator=FilterOperator.GTE, value=0.5)
        rel = MemoryRelationship(
            type="related_to",
            direction="in",
            node_type="Memory",
            condition=condition,
        )
        assert rel.condition is condition


class TestMemoryQuery:
    """Tests for MemoryQuery dataclass."""

    def test_defaults(self):
        """MemoryQuery has sensible defaults."""
        query = MemoryQuery()
        assert query.filters == []
        assert query.relationships == []
        assert query.vector is None
        assert query.limit == 10
        assert query.offset == 0
        assert query.include_raw is False
        assert query.sort_order == SortOrder.DESC

    def test_with_filters(self):
        """MemoryQuery can include filters."""
        filters = [
            MemoryFilter(field="content", operator=FilterOperator.CONTAINS, value="test"),
            MemoryFilter(field="importance", operator=FilterOperator.GTE, value=0.5),
        ]
        query = MemoryQuery(filters=filters)
        assert len(query.filters) == 2


class TestMemoryQueryBuilder:
    """Tests for MemoryQueryBuilder fluent API."""

    @pytest.mark.asyncio
    async def test_empty_builder(self):
        """Empty builder creates default query."""
        builder = MemoryQueryBuilder()
        query = await builder.build()
        assert isinstance(query, MemoryQuery)
        assert query.filters == []

    @pytest.mark.asyncio
    async def test_filter_method(self):
        """filter() adds filter to query."""
        query = await MemoryQueryBuilder().filter("importance", FilterOperator.GTE, 0.7).build()
        assert len(query.filters) == 1
        assert query.filters[0].field == "importance"
        assert query.filters[0].operator == FilterOperator.GTE
        assert query.filters[0].value == 0.7

    @pytest.mark.asyncio
    async def test_filter_with_string_operator(self):
        """filter() accepts string operator."""
        query = await MemoryQueryBuilder().filter("importance", "gte", 0.5).build()
        assert query.filters[0].operator == FilterOperator.GTE

    @pytest.mark.asyncio
    async def test_content_contains(self):
        """content_contains() adds CONTAINS filter on content."""
        query = await MemoryQueryBuilder().content_contains("important meeting").build()
        assert len(query.filters) == 1
        assert query.filters[0].field == "content"
        assert query.filters[0].operator == FilterOperator.CONTAINS

    @pytest.mark.asyncio
    async def test_content_matches(self):
        """content_matches() adds MATCHES filter for full-text search."""
        query = await MemoryQueryBuilder().content_matches("project deadline").build()
        assert len(query.filters) == 1
        assert query.filters[0].operator == FilterOperator.MATCHES

    @pytest.mark.asyncio
    async def test_similar_to_with_vector(self):
        """similar_to() can accept pre-computed vector."""
        vec = [0.1, 0.2, 0.3]
        query = await MemoryQueryBuilder().similar_to(vector=vec).build()
        assert query.vector == vec

    @pytest.mark.asyncio
    async def test_of_type_single(self):
        """of_type() with single type."""
        query = await MemoryQueryBuilder().of_type(MemoryType.SEMANTIC).build()
        assert len(query.filters) == 1
        assert query.filters[0].field == "memory_type"

    @pytest.mark.asyncio
    async def test_of_type_multiple(self):
        """of_type() with multiple types."""
        query = await MemoryQueryBuilder().of_type(MemoryType.SEMANTIC, MemoryType.EPISODIC).build()
        assert len(query.filters) == 1
        assert query.filters[0].operator == FilterOperator.IN
        assert MemoryType.SEMANTIC in query.filters[0].value
        assert MemoryType.EPISODIC in query.filters[0].value

    @pytest.mark.asyncio
    async def test_with_tags(self):
        """with_tags() adds tag filter."""
        query = await MemoryQueryBuilder().with_tags(["important", "urgent"]).build()
        assert len(query.filters) == 1
        assert query.filters[0].field == "tags"
        assert query.filters[0].operator == FilterOperator.CONTAINS

    @pytest.mark.asyncio
    async def test_with_emotions(self):
        """with_emotions() adds emotion filter."""
        query = await (
            MemoryQueryBuilder()
            .with_emotions([EmotionCategory.JOY, EmotionCategory.SURPRISE])
            .build()
        )
        assert len(query.filters) == 1
        assert query.filters[0].field == "emotions"

    @pytest.mark.asyncio
    async def test_importance_range_min_only(self):
        """importance_range() with min_value only."""
        query = await MemoryQueryBuilder().importance_range(min_value=0.5).build()
        assert len(query.filters) == 1
        assert query.filters[0].operator == FilterOperator.GTE

    @pytest.mark.asyncio
    async def test_importance_range_max_only(self):
        """importance_range() with max_value only."""
        query = await MemoryQueryBuilder().importance_range(max_value=0.8).build()
        assert len(query.filters) == 1
        assert query.filters[0].operator == FilterOperator.LTE

    @pytest.mark.asyncio
    async def test_importance_range_both(self):
        """importance_range() with both min and max."""
        query = await MemoryQueryBuilder().importance_range(min_value=0.3, max_value=0.9).build()
        assert len(query.filters) == 2

    @pytest.mark.asyncio
    async def test_in_timeframe(self):
        """in_timeframe() adds timestamp filters."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        query = await MemoryQueryBuilder().in_timeframe(start=start, end=end).build()
        assert len(query.filters) == 2

    @pytest.mark.asyncio
    async def test_sort_by(self):
        """sort_by() sets sort field and order."""
        query = await MemoryQueryBuilder().sort_by("importance", SortOrder.ASC).build()
        assert query.sort_by == "importance"
        assert query.sort_order == SortOrder.ASC

    @pytest.mark.asyncio
    async def test_sort_by_string_order(self):
        """sort_by() accepts string order."""
        query = await MemoryQueryBuilder().sort_by("timestamp", "asc").build()
        assert query.sort_order == SortOrder.ASC

    @pytest.mark.asyncio
    async def test_limit(self):
        """limit() sets result limit."""
        query = await MemoryQueryBuilder().limit(50).build()
        assert query.limit == 50

    @pytest.mark.asyncio
    async def test_offset(self):
        """offset() sets result offset."""
        query = await MemoryQueryBuilder().offset(20).build()
        assert query.offset == 20

    @pytest.mark.asyncio
    async def test_include_raw(self):
        """include_raw() sets include_raw flag."""
        query = await MemoryQueryBuilder().include_raw(True).build()
        assert query.include_raw is True

    @pytest.mark.asyncio
    async def test_chaining(self):
        """Builder methods can be chained."""
        query = await (
            MemoryQueryBuilder()
            .content_contains("meeting")
            .of_type(MemoryType.EPISODIC)
            .importance_range(min_value=0.5)
            .limit(20)
            .sort_by("timestamp")
            .build()
        )
        assert len(query.filters) == 3  # content, type, importance
        assert query.limit == 20
        assert query.sort_by == "timestamp"

    @pytest.mark.asyncio
    async def test_with_relationship(self):
        """with_relationship() adds graph relationship."""
        query = await (
            MemoryQueryBuilder()
            .with_relationship("related_to", "out", "Memory")
            .build()
        )
        assert len(query.relationships) == 1
        assert query.relationships[0].type == "related_to"

    @pytest.mark.asyncio
    async def test_add_pre_build_hook(self):
        """Pre-build hooks are executed on build."""
        hook_called = False

        async def test_hook(query: MemoryQuery):
            nonlocal hook_called
            hook_called = True

        builder = MemoryQueryBuilder().add_pre_build_hook(test_hook)
        query = await builder.build()

        assert hook_called
        assert isinstance(query, MemoryQuery)

    def test_filter_with_callback(self):
        """filter_with_callback() applies callback to builder."""

        def add_importance_filter(builder: MemoryQueryBuilder) -> MemoryQueryBuilder:
            return builder.filter("importance", FilterOperator.GTE, 0.5)

        builder = MemoryQueryBuilder().filter_with_callback(add_importance_filter)
        assert len(builder.filters) == 1
        assert builder.filters[0].field == "importance"
