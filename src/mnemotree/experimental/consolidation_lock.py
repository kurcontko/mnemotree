"""File-based consolidation lock with stale-PID recovery and mtime rollback.

Prevents concurrent consolidation runs across processes. The lock file's
mtime doubles as the last-successful-consolidation timestamp.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_FILENAME = ".consolidation-lock"


@dataclass
class LockState:
    """Snapshot of current lock state."""

    locked: bool
    holder_pid: int | None
    last_consolidated_at: datetime | None


class ConsolidationLock:
    """Process-safe file lock for memory consolidation.

    The lock file stores the holder PID as JSON.  Its mtime is used as
    the ``last_consolidated_at`` timestamp so that a successful run
    automatically advances the cooldown window.

    Stale-lock recovery: if the recorded PID is no longer running, the
    lock can be stolen by a new process.  On consolidation failure the
    mtime is rolled back so the next attempt isn't delayed.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._lock_path = self._dir / _LOCK_FILENAME
        self._pre_lock_mtime: float | None = None

    # -- public API -----------------------------------------------------------

    def state(self) -> LockState:
        """Read lock state without acquiring."""
        if not self._lock_path.exists():
            return LockState(locked=False, holder_pid=None, last_consolidated_at=None)
        try:
            data = json.loads(self._lock_path.read_text())
            pid = data.get("pid")
            mtime = self._lock_path.stat().st_mtime
            last_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
            alive = _pid_alive(pid) if pid else False
            return LockState(
                locked=alive, holder_pid=pid if alive else None, last_consolidated_at=last_at
            )
        except (json.JSONDecodeError, OSError):
            return LockState(locked=False, holder_pid=None, last_consolidated_at=None)

    def last_consolidated_at(self) -> datetime | None:
        """Return last successful consolidation time (lock file mtime)."""
        return self.state().last_consolidated_at

    def acquire(self) -> bool:
        """Try to acquire the lock.  Returns True on success."""
        self._dir.mkdir(parents=True, exist_ok=True)

        if self._lock_path.exists():
            try:
                data = json.loads(self._lock_path.read_text())
                holder_pid = data.get("pid")
            except (json.JSONDecodeError, OSError):
                holder_pid = None

            if holder_pid is not None and _pid_alive(holder_pid):
                logger.debug("Consolidation lock held by living PID %s", holder_pid)
                return False
            # Stale lock — steal it
            logger.info("Stealing stale consolidation lock (was PID %s)", holder_pid)

        # Save pre-lock mtime for potential rollback
        if self._lock_path.exists():
            self._pre_lock_mtime = self._lock_path.stat().st_mtime
        else:
            self._pre_lock_mtime = None

        self._lock_path.write_text(json.dumps({"pid": os.getpid()}))
        return True

    def release(self, *, success: bool = True) -> None:
        """Release the lock.  On failure, roll back mtime."""
        if not self._lock_path.exists():
            return

        if success:
            # Touch mtime to now — marks successful consolidation time
            self._lock_path.touch()
        else:
            # Roll back mtime so the cooldown doesn't advance
            if self._pre_lock_mtime is not None:
                try:
                    os.utime(self._lock_path, (self._pre_lock_mtime, self._pre_lock_mtime))
                except OSError:
                    logger.warning("Failed to roll back consolidation lock mtime")

        # Clear PID but keep the file (mtime is meaningful)
        try:
            self._lock_path.write_text(json.dumps({"pid": None}))
        except OSError:
            logger.warning("Failed to clear consolidation lock PID")

        self._pre_lock_mtime = None


def _pid_alive(pid: int) -> bool:
    """Check whether a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
