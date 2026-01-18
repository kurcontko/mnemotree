"""Shared serialization helpers for memory store implementations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def serialize_datetime(value: datetime | str | None) -> str | None:
    """Serialize a datetime value to ISO format string.

    Args:
        value: A datetime object, string, or None

    Returns:
        ISO format string, or None if value is None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def serialize_datetime_list(values: list[datetime | str]) -> list[str]:
    """Serialize a list of datetime values to ISO format strings.

    Args:
        values: List of datetime objects or strings

    Returns:
        List of ISO format strings
    """
    return [serialize_datetime(value) or "" for value in values]


def safe_load_context(context_str: str | None) -> dict[str, Any]:
    """Safely parse a JSON context string.

    Args:
        context_str: JSON string representing context, or None

    Returns:
        Parsed dict, or empty dict if parsing fails
    """
    if not context_str:
        return {}
    try:
        loaded = json.loads(context_str)
        return loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for comparison.

    Strips whitespace, lowercases, and removes leading articles (the, a, an).

    Args:
        text: Entity text to normalize

    Returns:
        Normalized entity text
    """
    normalized = text.strip().lower()
    for prefix in ("the ", "a ", "an "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].lstrip()
    return normalized


def normalize_filter_value(value: Any) -> Any:
    """Normalize a filter value for database queries.

    Converts datetime objects to ISO format strings, extracts Enum values,
    and recursively normalizes list elements. This ensures consistent
    filter value handling across all store backends.

    Args:
        value: The filter value to normalize (datetime, Enum, list, or other)

    Returns:
        Normalized value suitable for database queries:
        - datetime -> ISO format string
        - Enum -> enum.value
        - list -> recursively normalized list
        - other -> unchanged
    """
    from enum import Enum  # Import here to avoid circular import at module level

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [normalize_filter_value(v) for v in value]
    return value
