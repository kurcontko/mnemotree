"""Source-first ingestion and a single explicit retrieval pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .atoms import AtomHit, Extractor, extract_atoms
from .chunking import chunk_source
from .models import (
    Candidate,
    ChannelScore,
    Chunk,
    ContextSpan,
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
from .store import SQLiteEvidenceStore, now, vector_blob


def fuse(
    channels: dict[str, list[tuple[Chunk, float]]],
    *,
    weights: dict[str, float],
    k: int,
    ranks: dict[str, dict[str, int]] | None = None,
) -> list[Candidate]:
    by_id: dict[str, Chunk] = {}
    scores: dict[str, dict[str, ChannelScore]] = {}
    for channel, items in channels.items():
        for rank, (chunk, score) in enumerate(items, 1):
            by_id[chunk.id] = chunk
            source_rank = (ranks or {}).get(channel, {}).get(chunk.id, rank)
            scores.setdefault(chunk.id, {})[channel] = ChannelScore(rank=source_rank, score=score)
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
        extractor: Extractor | None = None,
        extraction_batch_chars: int = 12000,
    ) -> None:
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("require chunk_size > chunk_overlap >= 0")
        self.embedder = embedder
        if not 1 <= extraction_batch_chars <= 100000:
            raise ValueError("extraction_batch_chars must be between 1 and 100000")
        self.extractor = extractor
        self.extraction_batch_chars = extraction_batch_chars
        self.reranker = reranker
        self.counter = token_counter if token_counter is not None else TiktokenCounter()
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.store = SQLiteEvidenceStore(
            path,
            embedding_spec=embedder.spec if embedder else None,
            extractor_config={
                "extractor": extractor.spec.model_dump(),
                "batch_chars": extraction_batch_chars,
                "pipeline": "quoted-atoms-v1",
            }
            if extractor
            else None,
        )

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
        if self.store.extractor_config is not None and self.extractor is None:
            raise ValueError("writes to an atom database require its configured extractor")
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
        atoms = (
            await extract_atoms(
                self.extractor,
                sources,
                max_chars=self.extraction_batch_chars,
                signature=digest(self.store.extractor_config),
            )
            if self.extractor
            else []
        )
        atom_vectors = (
            await self.embedder.embed_documents([atom.text for atom in atoms])
            if self.embedder and atoms
            else None
        )
        return self.store.ingest(
            namespace,
            request_id,
            session_id,
            body_hash,
            sources,
            chunks,
            vectors,
            atoms,
            atom_vectors,
        )

    async def search(
        self, namespace: str, query: str, *, options: SearchOptions | None = None
    ) -> SearchResult:
        scope(namespace)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        options = options or SearchOptions(mode="hybrid" if self.embedder else "lexical")
        if options.representation != "raw" and self.store.extractor_config is None:
            raise ValueError("atom retrieval requires an extracted corpus")
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
        neighbors: dict[str, tuple[ContextSpan, ...]] = {}
        atom_hits: list[AtomHit] = []
        atom_ranks: dict[str, dict[str, int]] = {}
        # A single snapshot covers all channels and source materialization.
        # Concurrent writes cannot mix different versions in the returned evidence.
        with self.store.transaction(read=True):
            if options.representation != "atoms" and options.mode in ("lexical", "hybrid"):
                channels["lexical"] = self.store.lexical(namespace, query, options.candidate_limit)
            if options.representation != "atoms" and query_vector is not None:
                channels["dense"] = self.store.dense(
                    namespace, query_vector, options.candidate_limit
                )
            if options.representation != "raw":
                atom_channels = {}
                if options.mode in ("lexical", "hybrid"):
                    atom_channels["atom_lexical"] = self.store.atoms.lexical(
                        namespace, query, options.candidate_limit
                    )
                if query_vector is not None:
                    assert self.store.embedding_spec is not None
                    atom_channels["atom_dense"] = self.store.atoms.dense(
                        namespace,
                        vector_blob(query_vector, self.store.embedding_spec.dimensions),
                        options.candidate_limit,
                    )
                for channel, hits in atom_channels.items():
                    expanded: dict[str, tuple[Chunk, float]] = {}
                    atom_ranks[channel] = {}
                    for rank, (atom, score) in enumerate(hits, 1):
                        atom_hits.append(
                            AtomHit(atom=atom, channel=channel, rank=rank, score=score)
                        )
                        for chunk in self.store.atoms.chunks(namespace, atom):
                            expanded.setdefault(chunk.id, (chunk, score))
                            atom_ranks[channel].setdefault(chunk.id, rank)
                    channels[channel] = list(expanded.values())
            candidates = fuse(
                channels,
                weights={
                    "lexical": options.lexical_weight,
                    "dense": options.dense_weight,
                    "atom_lexical": options.lexical_weight * options.atom_weight,
                    "atom_dense": options.dense_weight * options.atom_weight,
                },
                k=options.rrf_k,
                ranks=atom_ranks,
            )
            for candidate in candidates:
                windows[candidate.chunk.id] = self.store.window(
                    namespace, candidate.chunk, options.neighbor_chunks
                )
                source = windows[candidate.chunk.id][0]
                neighbors[candidate.chunk.id] = self.store.conversation_context(
                    namespace,
                    source,
                    neighbors=options.neighbor_messages,
                    max_chars=options.neighbor_chars,
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
                    request_id=source.request_id if options.neighbor_messages else None,
                    position=source.position if options.neighbor_messages else None,
                    neighbors=neighbors[candidate.chunk.id],
                )
            )
        packing_start = perf_counter()
        selected, context, tokens, omitted = pack(
            evidence, self.counter, budget=options.token_budget, limit=options.limit
        )
        requested_context = {item.id: item for values in neighbors.values() for item in values}
        included_context = {item.id for entry in selected for item in entry.neighbors}
        return SearchResult(
            evidence=selected,
            context=context,
            trace=SearchTrace(
                mode=options.mode,
                candidates=tuple(candidates),
                atom_hits=tuple(hit.model_dump(mode="json") for hit in atom_hits),
                context_candidates=tuple(requested_context.values()),
                omitted_context_ids=tuple(
                    id for id in requested_context if id not in included_context
                ),
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
