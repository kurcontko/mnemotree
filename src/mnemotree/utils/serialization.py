from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel


def make_json_safe(value: Any) -> Any:
    """Convert arbitrary Python objects into JSON-serializable structures.

    This is intentionally conservative: unknown objects fall back to `str(value)`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, BaseModel):
        return make_json_safe(value.model_dump())

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    return str(value)


def json_dumps_safe(value: Any) -> str:
    """`json.dumps` that won't fail on non-serializable input."""
    return json.dumps(make_json_safe(value), ensure_ascii=False)


def json_loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
