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


def normalize_filter_value(value: Any) -> Any:
    if isinstance(value, MemoryType):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def build_sqlite_filter_clauses(
    filters: list[MemoryFilter],
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for filt in filters:
        field = filt.field
        if field not in {
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
        }:
            raise UnsupportedQueryError(f"Unsupported filter field: {field!r}")

        column = field
        operator = filt.operator
        raw_value = filt.value
        if isinstance(raw_value, list):
            values = [normalize_filter_value(item) for item in raw_value]
        else:
            values = [normalize_filter_value(raw_value)]

        if operator in {FilterOperator.EQ, FilterOperator.NE}:
            op = "=" if operator == FilterOperator.EQ else "!="
            clauses.append(f"{column} {op} ?")
            params.append(values[0])
            continue
        if operator in {
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
        }:
            op = {
                FilterOperator.GT: ">",
                FilterOperator.GTE: ">=",
                FilterOperator.LT: "<",
                FilterOperator.LTE: "<=",
            }[operator]
            clauses.append(f"{column} {op} ?")
            params.append(values[0])
            continue
        if operator in {FilterOperator.IN, FilterOperator.NOT_IN}:
            if not values:
                clauses.append(_SQL_FALSE if operator == FilterOperator.IN else _SQL_TRUE)
                continue
            placeholders = ", ".join("?" for _ in values)
            op = "IN" if operator == FilterOperator.IN else "NOT IN"
            clauses.append(f"{column} {op} ({placeholders})")
            params.extend(values)
            continue
        if operator in {FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS}:
            if field == "content":
                sub_clauses = []
                for item in values:
                    sub_clauses.append("content LIKE ?")
                    params.append(f"%{item}%")
                joined = " OR ".join(sub_clauses) if sub_clauses else _SQL_FALSE
                if operator == FilterOperator.NOT_CONTAINS:
                    clauses.append(f"NOT ({joined})")
                else:
                    clauses.append(f"({joined})")
                continue
            if field in _SQLITE_LIST_FIELDS:
                sub_clauses = []
                for item in values:
                    sub_clauses.append(f"(',' || {column} || ',') LIKE ?")
                    params.append(f"%,{item},%")
                joined = " OR ".join(sub_clauses) if sub_clauses else _SQL_FALSE
                if operator == FilterOperator.NOT_CONTAINS:
                    clauses.append(f"NOT ({joined})")
                else:
                    clauses.append(f"({joined})")
                continue
        if operator == FilterOperator.MATCHES:
            raise UnsupportedQueryError("Full-text MATCHES queries are not supported.")

        raise UnsupportedQueryError(
            f"Unsupported filter operator {operator!r} for field {field!r}."
        )
    return clauses, params
