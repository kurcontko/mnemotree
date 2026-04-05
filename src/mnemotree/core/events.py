"""Memory event system for observability and extensibility.

Emits ``MemoryEvent`` instances after core operations (remember, recall,
forget, link, consolidate).  Consumers register callbacks via
``MemoryCore.on_event()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    REMEMBER = "remember"
    RECALL = "recall"
    FORGET = "forget"
    LINK = "link"
    CONSOLIDATE = "consolidate"
    EVICT = "evict"


@dataclass
class MemoryEvent:
    """A single memory operation event."""

    event_type: EventType
    memory_ids: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Simple synchronous event dispatcher."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[MemoryEvent], Any]] = []

    def subscribe(self, callback: Callable[[MemoryEvent], Any]) -> None:
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[MemoryEvent], Any]) -> None:
        self._listeners.remove(callback)

    def emit(self, event: MemoryEvent) -> None:
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.warning(
                    "Event listener %s failed for %s",
                    getattr(listener, "__name__", listener),
                    event.event_type.value,
                    exc_info=True,
                )
