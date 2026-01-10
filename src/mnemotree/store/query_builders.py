from __future__ import annotations

from typing import Any

from ..core.query import FilterOperator, MemoryFilter
from .serialization import normalize_filter_value


class UnsupportedQueryError(NotImplementedError):
    """Raised when a MemoryQuery cannot be expressed for a specific backend."""


_NEO4J_LIST_FIELDS = {
    "tags",
    "emotions",
    "linked_concepts",
    "associations",
    "conflicts_with",
    "access_history",
}

_NEO4J_UNSUPPORTED_FIELDS = {
    # In the Neo4j store implementation these are modeled as relationships, not node properties.
    "tags",
    "associations",
    "conflicts_with",
}


def build_neo4j_where_clause(filters: list[MemoryFilter]) -> tuple[str, dict[str, Any]]:
    """Compile MemoryQuery filters into a Neo4j WHERE clause.

    This intentionally supports only a safe subset and fails fast for unsupported
    fields/operators to prevent silent mis-filtering.
    """
    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    op_map = {
        FilterOperator.EQ: "=",
        FilterOperator.NE: "<>",
        FilterOperator.GT: ">",
        FilterOperator.GTE: ">=",
        FilterOperator.LT: "<",
        FilterOperator.LTE: "<=",
    }

    for i, filter_cond in enumerate(filters):
        if filter_cond.field in _NEO4J_UNSUPPORTED_FIELDS:
            raise UnsupportedQueryError(
                f"Neo4j store does not support filtering by `{filter_cond.field}` "
                "via `query_memories()` (modeled as relationships)."
            )

        param_name = f"filter_{i}"
        field = f"m.{filter_cond.field}"
        value = normalize_filter_value(filter_cond.value)

        if filter_cond.operator in op_map:
            where_clauses.append(f"{field} {op_map[filter_cond.operator]} ${param_name}")
            params[param_name] = value
            continue

        if filter_cond.operator == FilterOperator.CONTAINS:
            if filter_cond.field in _NEO4J_LIST_FIELDS:
                where_clauses.append(f"${param_name} IN {field}")
                params[param_name] = value
                continue
            where_clauses.append(f"{field} CONTAINS ${param_name}")
            params[param_name] = value
            continue

        if filter_cond.operator == FilterOperator.NOT_CONTAINS:
            if filter_cond.field in _NEO4J_LIST_FIELDS:
                where_clauses.append(f"NOT (${param_name} IN {field})")
                params[param_name] = value
                continue
            where_clauses.append(f"NOT ({field} CONTAINS ${param_name})")
            params[param_name] = value
            continue

        if filter_cond.operator == FilterOperator.IN:
            if not isinstance(value, list):
                raise ValueError("FilterOperator.IN requires a list value.")
            where_clauses.append(f"{field} IN ${param_name}")
            params[param_name] = value
            continue

        if filter_cond.operator == FilterOperator.NOT_IN:
            if not isinstance(value, list):
                raise ValueError("FilterOperator.NOT_IN requires a list value.")
            where_clauses.append(f"NOT ({field} IN ${param_name})")
            params[param_name] = value
            continue

        if filter_cond.operator == FilterOperator.MATCHES:
            raise UnsupportedQueryError(
                "Neo4j full-text search (FilterOperator.MATCHES) is not supported by "
                "`query_memories()` yet."
            )

        raise UnsupportedQueryError(f"Unsupported operator for Neo4j: {filter_cond.operator}")

    if not where_clauses:
        return "", {}
    return "WHERE " + " AND ".join(where_clauses), params


def build_chroma_where(filters: list[MemoryFilter]) -> dict[str, str]:
    """Compile MemoryQuery filters into a Chroma where dict.

    Currently supports only equality filtering, because most metadata in the
    Chroma store is stored as strings and complex operators are not portable.
    """
    where: dict[str, str] = {}
    for f in filters:
        if f.operator != FilterOperator.EQ:
            raise UnsupportedQueryError(
                f"Chroma store only supports EQ filters; got {f.operator} for field {f.field!r}."
            )
        where[f.field] = str(normalize_filter_value(f.value))
    return where
