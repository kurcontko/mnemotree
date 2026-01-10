from __future__ import annotations

import time
from typing import Any


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def store_log_context(
    store_type: str,
    *,
    memory_id: str | None = None,
    duration_ms: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    context = {
        "store_type": store_type,
        "memory_id": memory_id,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }
    if extra:
        context.update(extra)
    return context
