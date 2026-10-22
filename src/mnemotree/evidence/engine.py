"""Source-first ingestion and a single explicit retrieval pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .chunking import chunk_source
from .models import (
    Candidate,
    ChannelScore,
    Chunk,
    Evidence,
    IngestReceipt,
    Message,
    SearchOptions,
    SearchResult,
    SearchTrace,
    Source,
    digest,
    scope,
)
from .packing import TiktokenCounter, pack
from .protocols import Embedder, Reranker, TokenCounter
from .store import SQLiteEvidenceStore, now


def fuse(
    channels: dict[str, list[tuple[Chunk, float]]], *, weights: dict[str, float], k: int
) -> list[Candidate]:
    by_id: dict[str, Chunk] = {}
    scores: dict[str, dict[str, ChannelScore]] = {}
    for channel, items in channels.items():
        for rank, (chunk, score) in enumerate(items, 1):
            by_id[chunk.id] = chunk
            scores.setdefault(chunk.id, {})[channel] = ChannelScore(rank=rank, score=score)
    candidates = [
        Candidate(
            chunk=by_id[id],
            channels=contributions,
            fusion_score=sum(
                weights[channel] / (k + value.rank) for channel, value in contributions.items()
            ),
        )
        for id, contributions in scores.items()
    ]
    return sorted(candidates, key=lambda candidate: (-candidate.fusion_score, candidate.chunk.id))


class Memory:
    """Local evidence memory, independent of legacy cognitive enrichment.

    Configure an embedder for dense/hybrid retrieval; otherwise the default is
    lexical-only. Explicit dense/hybrid requests never silently fall back.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        token_counter: TokenCounter | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("require chunk_size > chunk_overlap >= 0")
        self.embedder = embedder
        self.reranker = reranker
        self.counter = token_counter if token_counter is not None else TiktokenCounter()
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.store = SQLiteEvidenceStore(path, embedding_spec=embedder.spec if embedder else None)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def add(
        self,
        namespace: str,
        messages: Sequence[Message],
        *,
        request_id: str,
        session_id: str,
    ) -> IngestReceipt:
        scope(namespace)
        for value in (request_id, session_id):
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(
                    "request_id and session_id must be nonblank and at most 256 characters"
                )
        if not messages:
            raise ValueError("messages must not be empty")
        messages = tuple(Message.model_validate(message) for message in messages)
        body_hash = digest([session_id, [message.model_dump(mode="json") for message in messages]])
        replay = self.store.receipt(namespace, request_id, body_hash)
        if replay is not None:
            return replay
        if self.store.embedding_spec and self.embedder is None:
            raise ValueError("writes to a dense database require its configured embedder")
        ingested_at = now()
        sources = [
            Source(
                namespace=namespace,
                id=message.id if message.id is not None else digest([request_id, index]),
                request_id=request_id,
                session_id=session_id,
                position=index,
                content=message.content,
                role=message.role,
                timestamp=message.timestamp,
                ingested_at=ingested_at,
            )
            for index, message in enumerate(messages)
        ]
        chunks = [
            chunk
            for source in sources
            for chunk in chunk_source(
                source.id, source.content, size=self.chunk_size, overlap=self.chunk_overlap
            )
        ]
        # Network/provider work is outside the write transaction. A failure leaves
        # no half-ingested sources or acknowledged-but-unindexed memories.
        vectors = (
            await self.embedder.embed_documents([chunk.text for chunk in chunks])
            if self.embedder
            else None
        )
        return self.store.ingest(
            namespace, request_id, session_id, body_hash, sources, chunks, vectors
        )

    async def search(
        self, namespace: str, query: str, *, options: SearchOptions | None = None
    ) -> SearchResult:
        scope(namespace)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        options = options or SearchOptions(mode="hybrid" if self.embedder else "lexical")
        started = perf_counter()
        query_vector = None
        if options.mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise ValueError("dense/hybrid retrieval requires the indexed embedder")
            query_vector = await self.embedder.embed_query(query)
        embedding_ms = (perf_counter() - started) * 1000
        retrieval_start = perf_counter()
        channels: dict[str, list[tuple[Chunk, float]]] = {}
        windows: dict[str, tuple[Source, int, int]] = {}
        # A single snapshot covers all channels and source materialization.
        # Concurrent writes cannot mix different versions in the returned evidence.
        with self.store.transaction(read=True):
            if options.mode in ("lexical", "hybrid"):
                channels["lexical"] = self.store.lexical(namespace, query, options.candidate_limit)
            if query_vector is not None:
                channels["dense"] = self.store.dense(
                    namespace, query_vector, options.candidate_limit
                )
            candidates = fuse(
                channels,
                weights={"lexical": options.lexical_weight, "dense": options.dense_weight},
                k=options.rrf_k,
            )
            for candidate in candidates:
                windows[candidate.chunk.id] = self.store.window(
                    namespace, candidate.chunk, options.neighbor_chunks
                )
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        rerank_start = perf_counter()
        if self.reranker and candidates:
            scores = list(
                await self.reranker.score(query, [item.chunk.text for item in candidates])
            )
            if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
                raise ValueError("reranker must return one finite score per candidate")
            candidates = [
                item.model_copy(update={"rerank_score": float(score)})
                for item, score in zip(candidates, scores, strict=True)
            ]
            candidates.sort(
                key=lambda item: (-(item.rerank_score or 0.0), -item.fusion_score, item.chunk.id)
            )
        rerank_ms = (perf_counter() - rerank_start) * 1000
        evidence = []
        for candidate in candidates:
            source, start, end = windows[candidate.chunk.id]
            evidence.append(
                Evidence(
                    id=candidate.chunk.id,
                    source_id=source.id,
                    session_id=source.session_id,
                    role=source.role,
                    timestamp=source.timestamp,
                    start=start,
                    end=end,
                    text=source.content[start:end],
                    score=candidate.fusion_score,
                    channels=candidate.channels,
                    rerank_score=candidate.rerank_score,
                )
            )
        packing_start = perf_counter()
        selected, context, tokens, omitted = pack(
            evidence, self.counter, budget=options.token_budget, limit=options.limit
        )
        return SearchResult(
            evidence=selected,
            context=context,
            trace=SearchTrace(
                mode=options.mode,
                candidates=tuple(candidates),
                channel_counts={channel: len(items) for channel, items in channels.items()},
                packed_tokens=tokens,
                tokenizer=self.counter.signature,
                omitted_ids=omitted,
                timings_ms={
                    "embedding": embedding_ms,
                    "retrieval": retrieval_ms,
                    "reranking": rerank_ms,
                    "packing": (perf_counter() - packing_start) * 1000,
                    "total": (perf_counter() - started) * 1000,
                },
            ),
        )

    def source(self, namespace: str, source_id: str) -> Source | None:
        return self.store.get_source(namespace, source_id)

    def forget(self, namespace: str, source_id: str) -> bool:
        return self.store.forget(namespace, source_id)
