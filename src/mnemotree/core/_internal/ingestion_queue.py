from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from ..models import MemoryType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionRequest:
    memory_id: str
    content: str
    memory_type: MemoryType | None
    importance: float | None
    tags: list[str] | None
    context: dict[str, object] | None
    analyze: bool | None
    summarize: bool | None
    references: list[str] | None
    timestamp: datetime | None


class MemoryIngestionQueue:
    def __init__(
        self,
        handler: Callable[[IngestionRequest], Awaitable[None]],
        *,
        maxsize: int = 100,
    ) -> None:
        self._handler = handler
        self._queue: asyncio.Queue[IngestionRequest] = asyncio.Queue(maxsize=maxsize)
        self._shutdown = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def enqueue(self, request: IngestionRequest) -> None:
        if self._shutdown.is_set():
            raise RuntimeError("Ingestion queue is shutting down.")
        await self._queue.put(request)

    async def shutdown(self, *, drain: bool = True) -> None:
        self._shutdown.set()
        if drain:
            await self._queue.join()
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        self._worker_task = None

    async def _worker_loop(self) -> None:
        while True:
            if self._shutdown.is_set() and self._queue.empty():
                break
            request = await self._queue.get()
            try:
                await self._handler(request)
            except Exception:
                logger.exception("Ingestion worker failed for memory %s", request.memory_id)
            finally:
                self._queue.task_done()
