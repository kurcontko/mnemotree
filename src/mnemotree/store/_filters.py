from __future__ import annotations

from datetime import datetime
from typing import Any

from ..core.models import MemoryType
from ..core.query import FilterOperator, MemoryFilter
from .query_builders import UnsupportedQueryError

_SQLITE_LIST_FIELDS = {
    "tags",
    "emotions",
    "associations",
    "linked_concepts",
    "conflicts_with",
}

_SQL_FALSE = "1 = 0"
_SQL_TRUE = "1 = 1"
_SUPPORTED_FIELDS = {
    "memory_id",
    "content",
    "memory_type",
    "timestamp",
    "importance",
    "confidence",
    "source",
    "tags",
    "emotions",
    "associations",
    "linked_concepts",
    "conflicts_with",
    "previous_event_id",
    "next_event_id",
}


def normalize_filter_value(value: Any) -> Any:
    if isinstance(value, MemoryType):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _validate_filter_field(field: str) -> str:
    if field not in _SUPPORTED_FIELDS:
        raise UnsupportedQueryError(f"Unsupported filter field: {field!r}")
    return field


def _normalize_filter_values(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, list):
        return [normalize_filter_value(item) for item in raw_value]
    return [normalize_filter_value(raw_value)]


def _wrap_contains_clause(joined: str, operator: FilterOperator) -> str:
    if operator == FilterOperator.NOT_CONTAINS:
        return f"NOT ({joined})"
    return f"({joined})"


def _build_simple_clause(
    column: str,
    operator: FilterOperator,
    value: Any,
    params: list[Any],
) -> str:
    op = "=" if operator == FilterOperator.EQ else "!="
    params.append(value)
    return f"{column} {op} ?"


def _build_comparison_clause(
    column: str,
    operator: FilterOperator,
    value: Any,
    params: list[Any],
) -> str:
    op = {
        FilterOperator.GT: ">",
        FilterOperator.GTE: ">=",
        FilterOperator.LT: "<",
        FilterOperator.LTE: "<=",
    }[operator]
    params.append(value)
    return f"{column} {op} ?"


def _build_in_clause(
    column: str,
    operator: FilterOperator,
    values: list[Any],
    params: list[Any],
) -> str:
    if not values:
        return _SQL_FALSE if operator == FilterOperator.IN else _SQL_TRUE
    placeholders = ", ".join("?" for _ in values)
    op = "IN" if operator == FilterOperator.IN else "NOT IN"
    params.extend(values)
    return f"{column} {op} ({placeholders})"


def _build_contains_clause(
    field: str,
    column: str,
    operator: FilterOperator,
    values: list[Any],
    params: list[Any],
) -> str | None:
    if field == "content":
        sub_clauses = []
        for item in values:
            sub_clauses.append("content LIKE ?")
            params.append(f"%{item}%")
        joined = " OR ".join(sub_clauses) if sub_clauses else _SQL_FALSE
        return _wrap_contains_clause(joined, operator)
    if field in _SQLITE_LIST_FIELDS:
        sub_clauses = []
        for item in values:
            sub_clauses.append(f"(',' || {column} || ',') LIKE ?")
            params.append(f"%,{item},%")
        joined = " OR ".join(sub_clauses) if sub_clauses else _SQL_FALSE
        return _wrap_contains_clause(joined, operator)
    return None


def _build_filter_clause(
    filt: MemoryFilter,
    params: list[Any],
) -> str:
    field = _validate_filter_field(filt.field)
    column = field
    operator = filt.operator
    values = _normalize_filter_values(filt.value)

    if operator in {FilterOperator.EQ, FilterOperator.NE}:
        return _build_simple_clause(column, operator, values[0], params)
    if operator in {
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    }:
        return _build_comparison_clause(column, operator, values[0], params)
    if operator in {FilterOperator.IN, FilterOperator.NOT_IN}:
        return _build_in_clause(column, operator, values, params)
    if operator in {FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS}:
        clause = _build_contains_clause(field, column, operator, values, params)
        if clause is not None:
            return clause
    if operator == FilterOperator.MATCHES:
        raise UnsupportedQueryError("Full-text MATCHES queries are not supported.")

    raise UnsupportedQueryError(
        f"Unsupported filter operator {operator!r} for field {field!r}."
    )


def build_sqlite_filter_clauses(
    filters: list[MemoryFilter],
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for filt in filters:
        clauses.append(_build_filter_clause(filt, params))
    return clauses, params
