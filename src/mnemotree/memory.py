"""Namespace-bound memory verbs over the Evidence engine.

`Memory` binds one namespace and speaks remember/recall/get/forget. Every call
goes through `mnemotree.evidence.Memory`, so ingestion stays transactional,
search stays read-only, and recall returns the engine's packed context and trace.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .evidence import engine
from .evidence.models import (
    Evidence,
    Message,
    Record,
    SearchOptions,
    SearchTrace,
    Source,
    digest,
    scope,
)
from .evidence.protocols import Embedder, Reranker, TokenCounter


class MemoryItem(Record):
    """One remembered message, exactly as it was stored."""

    id: str
    content: str
    role: str
    session: str
    timestamp: datetime | None
    created_at: datetime

    @classmethod
    def from_source(cls, source: Source) -> MemoryItem:
        return cls(
            id=source.id,
            content=source.content,
            role=source.role,
            session=source.session_id,
            timestamp=source.timestamp,
            created_at=source.ingested_at,
        )


class Recall(Record):
    """Packed evidence for a query, with the memories it came from."""

    # Distinct sources of the packed evidence, including neighboring turns.
    items: tuple[MemoryItem, ...]
    evidence: tuple[Evidence, ...]
    context: str
    trace: SearchTrace


class Memory:
    """Persistent memory for one namespace.

    Without an embedder retrieval is lexical; with one it defaults to hybrid.
    Pass ``search`` to fix the default search options for every recall.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        token_counter: TokenCounter | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        search: SearchOptions | None = None,
    ) -> None:
        self.namespace = scope(namespace)
        self.search = search
        self.engine = engine.Memory(
            path,
            embedder=embedder,
            reranker=reranker,
            token_counter=token_counter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @classmethod
    def from_engine(
        cls, memory: engine.Memory, *, namespace: str, search: SearchOptions | None = None
    ) -> Memory:
        """Bind an already configured engine, such as one built from a CLI config.

        The returned Memory owns the engine: closing it closes the engine.
        """
        self = cls.__new__(cls)
        self.namespace = scope(namespace)
        self.search = search
        self.engine = memory
        return self

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def remember(
        self,
        content: str,
        *,
        role: str = "user",
        session: str | None = None,
        timestamp: datetime | None = None,
        id: str | None = None,
        request_id: str | None = None,
    ) -> MemoryItem:
        """Store one message. Pass ``request_id`` to make retries idempotent.

        Without a ``session`` the message is its own session, so it never
        becomes a neighbouring turn of unrelated memories.
        """
        message = Message(content=content, role=role, id=id, timestamp=timestamp)
        request_id = request_id or uuid4().hex
        items = await self.remember_many(
            [message], session=session or digest(["isolated", request_id]), request_id=request_id
        )
        return items[0]

    async def remember_many(
        self, messages: Sequence[Message], *, session: str, request_id: str | None = None
    ) -> list[MemoryItem]:
        """Store consecutive turns of one session, in order, as one atomic write."""
        receipt = await self.engine.add(
            self.namespace,
            messages,
            request_id=request_id or uuid4().hex,
            session_id=session,
        )
        sources = await asyncio.to_thread(self.engine.sources, self.namespace, receipt.source_ids)
        missing = [id for id in receipt.source_ids if id not in sources]
        if missing:
            raise LookupError(f"memories were forgotten concurrently: {', '.join(missing)}")
        return [MemoryItem.from_source(sources[id]) for id in receipt.source_ids]

    async def recall(
        self, query: str, *, limit: int | None = None, options: SearchOptions | None = None
    ) -> Recall:
        """Retrieve and pack evidence for ``query``. Never changes stored state."""
        options = options or self.search
        if limit is not None:
            base = options or SearchOptions(mode="hybrid" if self.engine.embedder else "lexical")
            deep = base.deep_candidate_limit
            options = SearchOptions.model_validate(
                {
                    **base.model_dump(),
                    "limit": limit,
                    "candidate_limit": max(limit, base.candidate_limit),
                    "deep_candidate_limit": None if deep is None else max(limit, deep),
                }
            )
        result = await self.engine.search(self.namespace, query, options=options)
        return Recall(
            items=tuple(MemoryItem.from_source(source) for source in result.sources),
            evidence=result.evidence,
            context=result.context,
            trace=result.trace,
        )

    async def get(self, id: str) -> MemoryItem | None:
        source = await asyncio.to_thread(self.engine.source, self.namespace, id)
        return MemoryItem.from_source(source) if source is not None else None

    async def forget(self, id: str) -> bool:
        """Delete a memory with its chunks and index entries. True if it existed."""
        return await asyncio.to_thread(self.engine.forget, self.namespace, id)
