"""Automatic consolidation scheduling with gate chain.

Wraps ``MemoryConsolidator`` with configurable triggers so consolidation
runs only when meaningful work has accumulated:

1. Minimum hours since last consolidation
2. Minimum memories added since last consolidation
3. File-based lock acquisition (prevents concurrent runs)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .consolidation import ConsolidationConfig, ConsolidationResult
from .consolidation_lock import ConsolidationLock

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".consolidation-state.json"


@dataclass
class SchedulerConfig:
    """Configuration for automatic consolidation triggers."""

    # Gate thresholds
    min_hours_since_last: float = 24.0
    min_memories_since_last: int = 50

    # How often to check (every N-th remember() call)
    check_interval: int = 20

    # Consolidation parameters (forwarded to ConsolidationConfig)
    consolidation_config: ConsolidationConfig = field(default_factory=ConsolidationConfig)


@dataclass
class SchedulerState:
    """Persisted scheduler state."""

    memories_since_last: int = 0
    last_checked_at: float = 0.0


class ConsolidationScheduler:
    """Gate-based automatic consolidation scheduler.

    Usage::

        scheduler = ConsolidationScheduler(store_dir="/path/to/store")

        # Call after each remember() — cheap counter bump, runs consolidation
        # only when all gates pass
        await scheduler.notify_remember()

        # Or check explicitly
        result = await scheduler.maybe_consolidate(memory_core)
    """

    def __init__(
        self,
        store_dir: str | Path,
        config: SchedulerConfig | None = None,
    ) -> None:
        self._dir = Path(store_dir)
        self._config = config or SchedulerConfig()
        self._lock = ConsolidationLock(self._dir)
        self._state_path = self._dir / _STATE_FILENAME
        self._call_count = 0
        self._state = self._load_state()

    def notify_remember(self) -> None:
        """Bump the memory counter.  Called after each remember()."""
        self._state.memories_since_last += 1
        self._call_count += 1
        # Persist periodically, not on every call
        if self._call_count % self._config.check_interval == 0:
            self._save_state()

    def should_consolidate(self) -> tuple[bool, str]:
        """Check all gates without acquiring the lock.

        Returns:
            (should_run, reason) — reason explains why not if False.
        """
        if self._call_count % self._config.check_interval != 0:
            return False, "not at check interval"

        # Gate 1: enough memories accumulated
        if self._state.memories_since_last < self._config.min_memories_since_last:
            return False, (
                f"only {self._state.memories_since_last} memories since last "
                f"(need {self._config.min_memories_since_last})"
            )

        # Gate 2: enough time elapsed
        last_at = self._lock.last_consolidated_at()
        if last_at is not None:
            hours_elapsed = (datetime.now(timezone.utc) - last_at).total_seconds() / 3600
            if hours_elapsed < self._config.min_hours_since_last:
                return False, (
                    f"only {hours_elapsed:.1f}h since last consolidation "
                    f"(need {self._config.min_hours_since_last}h)"
                )

        return True, "all gates passed"

    async def maybe_consolidate(self, memory_core: Any) -> ConsolidationResult | None:
        """Run consolidation if all gates pass, including lock acquisition.

        Args:
            memory_core: A ``MemoryCore`` instance.

        Returns:
            ``ConsolidationResult`` if consolidation ran, ``None`` otherwise.
        """
        should, reason = self.should_consolidate()
        if not should:
            logger.debug("Skipping consolidation: %s", reason)
            return None

        # Gate 3: acquire lock
        if not self._lock.acquire():
            logger.debug("Skipping consolidation: lock held by another process")
            return None

        success = False
        try:
            result = await memory_core.consolidate(
                config=self._config.consolidation_config,
            )
            success = True

            # Reset counters on success
            self._state.memories_since_last = 0
            self._state.last_checked_at = time.time()
            self._save_state()

            logger.info(
                "Auto-consolidation complete: %d clusters, %d semantic created, %d deprecated",
                result.clusters_formed,
                result.semantic_memories_created,
                result.memories_deprecated,
            )
            return result
        finally:
            self._lock.release(success=success)

    def _load_state(self) -> SchedulerState:
        if not self._state_path.exists():
            return SchedulerState()
        try:
            data = json.loads(self._state_path.read_text())
            return SchedulerState(
                memories_since_last=data.get("memories_since_last", 0),
                last_checked_at=data.get("last_checked_at", 0.0),
            )
        except (json.JSONDecodeError, OSError):
            return SchedulerState()

    def _save_state(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self._state_path.write_text(
                json.dumps(
                    {
                        "memories_since_last": self._state.memories_since_last,
                        "last_checked_at": self._state.last_checked_at,
                    }
                )
            )
        except OSError:
            logger.warning("Failed to save consolidation scheduler state")
