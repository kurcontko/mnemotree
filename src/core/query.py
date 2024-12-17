from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Awaitable

from pydantic import BaseModel, Field

from ..core.models import MemoryType, EmotionCategory


class FilterOperator(str, Enum):
    """Operators for memory filters."""
    EQ = "eq"         # Equal
    NE = "ne"         # Not equal
    GT = "gt"         # Greater than
    GTE = "gte"       # Greater than or equal
    LT = "lt"         # Less than
    LTE = "lte"       # Less than or equal
    IN = "in"         # In list
    NOT_IN = "not_in" # Not in list
    CONTAINS = "contains"         # Contains string/element
    NOT_CONTAINS = "not_contains" # Does not contain
    MATCHES = "matches" # For full text search


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class MemoryFilter:
    """Single filter condition for memory queries."""
    field: str
    operator: FilterOperator
    value: Any


@dataclass
class MemoryRelationship:
  """Graph relationship for memory queries."""
  type: str
  direction: str # out/in/any
  node_type: str
  condition: Optional[MemoryFilter] = None

@dataclass
class MemoryQuery:
    """
    Memory query specification.
    
    Examples:
        query = MemoryQuery(
            filters=[
                MemoryFilter("content", FilterOperator.CONTAINS, "project meeting"),
                MemoryFilter("memory_type", FilterOperator.IN, [MemoryType.EPISODIC]),
                MemoryFilter("tags", FilterOperator.CONTAINS, ["important"]),
                MemoryFilter("importance", FilterOperator.GTE, 0.7)
            ],
            vector=[0.1, 0.2, ...],
            sort_by="timestamp",
            sort_order=SortOrder.DESC
        )
    """
    # Main query components
    filters: List[MemoryFilter] = field(default_factory=list)
    relationships: List[MemoryRelationship] = field(default_factory=list)
    vector: Optional[List[float]] = None
    
    # Query settings
    limit: int = 10
    offset: int = 0
    include_raw: bool = False
    
    # Sorting
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC


class MemoryQueryBuilder:
    """
    Builder for memory queries.
    
    Examples:
        query = (MemoryQueryBuilder()
                .content_contains("project meeting")
                .with_tags(["important"])
                .min_importance(0.7)
                .sort_by("timestamp", "desc")
                .limit(5)
                .build())
    """
    
    def __init__(self):
        self.filters: List[MemoryFilter] = []
        self.relationships: List[MemoryRelationship] = []
        self.vector: Optional[List[float]] = None
        self.limit_val: int = 10
        self.offset_val: int = 0
        self.include_raw_val: bool = False
        self.sort_by_val: Optional[str] = None
        self.sort_order_val: SortOrder = SortOrder.DESC
        
        # Placeholders for database-specific functions
        self._pre_build_hooks: List[Callable[["MemoryQuery"], Awaitable[None]]] = []

    
    def filter(
        self,
        field: str,
        operator: Union[FilterOperator, str],
        value: Any
    ) -> "MemoryQueryBuilder":
        """Add a filter condition."""
        if isinstance(operator, str):
            operator = FilterOperator(operator)
        
        self.filters.append(MemoryFilter(field, operator, value))
        return self

    def filter_with_callback(
      self, 
      callback: Callable[["MemoryQueryBuilder"], "MemoryQueryBuilder"]
      ) -> "MemoryQueryBuilder":
        """Add filter with callback function"""
        return callback(self)


    def content_contains(self, text: str) -> "MemoryQueryBuilder":
        """Filter by content text."""
        return self.filter("content", FilterOperator.CONTAINS, text)

    def content_matches(self, text: str) -> "MemoryQueryBuilder":
        """Filter by content text using full-text search."""
        return self.filter("content", FilterOperator.MATCHES, text)
    
    def similar_to(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None
    ) -> "MemoryQueryBuilder":
        """Find similar memories by text or vector."""
        if vector is not None:
            self.vector = vector
        if text is not None:
            self.content_contains(text)
        return self
    
    def of_type(self, *types: MemoryType) -> "MemoryQueryBuilder":
        """Filter by memory types."""
        return self.filter("memory_type", FilterOperator.IN, list(types))
    
    def with_tags(self, tags: List[str]) -> "MemoryQueryBuilder":
        """Filter by tags."""
        return self.filter("tags", FilterOperator.CONTAINS, tags)
    
    def with_emotions(self, emotions: List[EmotionCategory]) -> "MemoryQueryBuilder":
        """Filter by emotions."""
        return self.filter("emotions", FilterOperator.CONTAINS, emotions)

    def with_relationship(
      self,
      relationship_type: str,
      direction: str, # 'out', 'in', 'any'
      node_type: str,
      condition: Optional[MemoryFilter] = None
    ) -> "MemoryQueryBuilder":
      """Filter based on graph relationships."""
      self.relationships.append(MemoryRelationship(relationship_type, direction, node_type, condition))
      return self
    
    def importance_range(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> "MemoryQueryBuilder":
        """Filter by importance range."""
        if min_value is not None:
            self.filter("importance", FilterOperator.GTE, min_value)
        if max_value is not None:
            self.filter("importance", FilterOperator.LTE, max_value)
        return self
    
    def in_timeframe(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> "MemoryQueryBuilder":
        """Filter by time range."""
        if start:
            self.filter("timestamp", FilterOperator.GTE, start)
        if end:
            self.filter("timestamp", FilterOperator.LTE, end)
        return self
    
    def sort_by(
        self,
        field: str,
        order: Union[SortOrder, str] = SortOrder.DESC
    ) -> "MemoryQueryBuilder":
        """Set sort order."""
        if isinstance(order, str):
            order = SortOrder(order)
        
        self.sort_by_val = field
        self.sort_order_val = order
        return self
    
    def limit(self, limit: int) -> "MemoryQueryBuilder":
        """Set result limit."""
        self.limit_val = limit
        return self
    
    def offset(self, offset: int) -> "MemoryQueryBuilder":
        """Set result offset."""
        self.offset_val = offset
        return self
    
    def include_raw(self, include: bool = True) -> "MemoryQueryBuilder":
        """Include raw content in results."""
        self.include_raw_val = include
        return self
    
    def add_pre_build_hook(self, hook: Callable[["MemoryQuery"], Awaitable[None]]) -> "MemoryQueryBuilder":
        """Register a hook that will be executed on pre build."""
        self._pre_build_hooks.append(hook)
        return self
    
    async def build(self) -> MemoryQuery:
        """Build the query."""
        query =  MemoryQuery(
            filters=self.filters,
            relationships = self.relationships,
            vector=self.vector,
            limit=self.limit_val,
            offset=self.offset_val,
            include_raw=self.include_raw_val,
            sort_by=self.sort_by_val,
            sort_order=self.sort_order_val
        )
        for hook in self._pre_build_hooks:
          await hook(query)
        return query


# Usage examples:
"""
# Simple content and tag query
query = (MemoryQueryBuilder()
         .content_contains("project meeting")
         .with_tags(["important"])
         .build())

# Vector similarity with filters
query = (MemoryQueryBuilder()
         .similar_to(vector=[0.1, 0.2, ...])
         .importance_range(min_value=0.7)
         .with_tags(["feedback"])
         .sort_by("importance", "desc")
         .build())

# Advanced filtering
query = (MemoryQueryBuilder()
         .filter("access_count", FilterOperator.GTE, 5)
         .filter("emotional_valence", FilterOperator.GT, 0.5)
         .in_timeframe(start_date, end_date)
         .of_type(MemoryType.EPISODIC, MemoryType.SEMANTIC)
         .build())

# Graph relationships
query = (MemoryQueryBuilder()
         .with_relationship("HAS_AUTHOR", "out", "User", MemoryFilter("name", FilterOperator.CONTAINS, "John"))
         .with_relationship("SIMILAR_TO", "in", "Memory", MemoryFilter("importance", FilterOperator.GTE, 0.8))
         .build())


#Full text Search
query = (MemoryQueryBuilder()
            .content_matches("project* AND meeting")
            .build())


# Async query with database hook
async def custom_database_hook(query: MemoryQuery):
  # Apply specific logic for database dialect, here for elasticsearch
    query.filters = [
    MemoryFilter(field="content", operator=FilterOperator.MATCHES, value=" ".join(filter.value.split(' AND '))) if filter.field == "content" and filter.operator == FilterOperator.MATCHES else filter for filter in query.filters
  ]

async def main():
    query = (MemoryQueryBuilder()
            .content_matches("project* AND meeting")
            .add_pre_build_hook(custom_database_hook)
            .build())
"""