"""Source-first ingestion and a single explicit retrieval pipeline."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .core.models import (
    Candidate,
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
    identifier,
    scope,
)
from .core.protocols import Embedder, Reranker, TokenCounter
from .errors import ConfigurationError, InvalidInput
from .ingest.atoms import Atom, AtomHit, Extractor, extract_atoms
from .ingest.chunking import chunk_source
from .retrieval.packing import Packed, TiktokenCounter, pack

# Re-exported: these lived here before ranking.py held the pure ranking steps.
from .retrieval.ranking import QUERY_EMBED_CHARS as QUERY_EMBED_CHARS
from .retrieval.ranking import QUERY_EMBED_TOKENS as QUERY_EMBED_TOKENS
from .retrieval.ranking import (
    RolePolicy,
    boost_role,
    by_rerank,
    cap_per_source,
    channel_weights,
    chronological,
    fuse,
    model_query,
    role_first,
    role_policy,
)
from .storage.store import SQLiteEvidenceStore, Window, now, vector_blob

__all__ = [
    "QUERY_EMBED_CHARS",
    "QUERY_EMBED_TOKENS",
    "Memory",
    "cap_per_source",
    "chronological",
    "fuse",
    "model_query",
    "role_first",
]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """The query as each channel receives it."""

    # Lexical channels search the question and its options as given.
    text: str
    options: Sequence[str] | None
    # Embedding and reranking receive the bounded model input.
    model_input: str
    original_tokens: int
    tokens: int
    truncated: bool
    vector: Sequence[float] | None


@dataclass(slots=True)
class Retrieval:
    """Everything one search reads from a single store snapshot."""

    candidate_limit: int
    chunk_density: float | None
    assistant_share: float | None
    channels: dict[str, list[tuple[Chunk, float]]] = field(default_factory=dict)
    atom_hits: list[AtomHit] = field(default_factory=list)
    # Fused anchors, and chunks only placed into budget the anchors leave unused.
    candidates: list[Candidate] = field(default_factory=list)
    overflow: list[Candidate] = field(default_factory=list)
    windows: dict[str, Window] = field(default_factory=dict)
    neighbors: dict[str, tuple[ContextSpan, ...]] = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)

    def roles(self) -> dict[str, str]:
        """Role of each retrieved chunk's source, by chunk ID."""
        return {id: window.source.role for id, window in self.windows.items()}


def to_evidence(
    candidate: Candidate,
    window: Window,
    context: tuple[ContextSpan, ...],
    *,
    with_position: bool,
) -> Evidence:
    source, start, end = window
    return Evidence(
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
        request_id=source.request_id if with_position else None,
        position=source.position if with_position else None,
        seq=source.session_seq,
        neighbors=context,
    )


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
            raise ConfigurationError("require chunk_size > chunk_overlap >= 0")
        self.embedder = embedder
        if not 1 <= extraction_batch_chars <= 100000:
            raise ConfigurationError("extraction_batch_chars must be between 1 and 100000")
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
        identifier(request_id, "request_id")
        identifier(session_id, "session_id")
        if not messages:
            raise InvalidInput("messages must not be empty")
        messages = tuple(Message.model_validate(message) for message in messages)
        body_hash = digest([session_id, [message.model_dump(mode="json") for message in messages]])
        replay = await asyncio.to_thread(self.store.receipt, namespace, request_id, body_hash)
        if replay is not None:
            return replay
        if self.store.embedding_spec and self.embedder is None:
            raise ConfigurationError("writes to a dense database require its configured embedder")
        if self.store.extractor_config is not None and self.extractor is None:
            raise ConfigurationError("writes to an atom database require its configured extractor")
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
        vectors, atoms = await self._embed_and_extract(sources, chunks)
        atom_vectors = (
            await self.embedder.embed_documents([atom.text for atom in atoms])
            if self.embedder and atoms
            else None
        )
        return await asyncio.to_thread(
            self.store.ingest,
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

    async def _embed_and_extract(
        self, sources: Sequence[Source], chunks: Sequence[Chunk]
    ) -> tuple[Sequence[Sequence[float]] | None, list[Atom]]:
        """Embed the chunks and extract atoms; with both providers, concurrently."""
        embedder, extractor = self.embedder, self.extractor
        texts = [chunk.text for chunk in chunks]
        if embedder is None or extractor is None:
            vectors = await embedder.embed_documents(texts) if embedder else None
            atoms = await self._extract(extractor, sources) if extractor else []
            return vectors, atoms
        try:
            async with asyncio.TaskGroup() as group:
                embedding = group.create_task(embedder.embed_documents(texts))
                extraction = group.create_task(self._extract(extractor, sources))
        except ExceptionGroup:
            # Raise the provider's own exception, as the sequential path does; a
            # group would hide it from callers that catch its type.
            errors = [
                error
                for task in (embedding, extraction)
                if task.done() and not task.cancelled() and (error := task.exception()) is not None
            ]
        else:
            return embedding.result(), extraction.result()
        # Raised outside the handler so the error keeps its own cause and context.
        raise errors[0]

    async def _extract(self, extractor: Extractor, sources: Sequence[Source]) -> list[Atom]:
        return await extract_atoms(
            extractor,
            sources,
            max_chars=self.extraction_batch_chars,
            signature=digest(self.store.extractor_config),
        )

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        options: SearchOptions | None = None,
        query_options: Sequence[str] | None = None,
    ) -> SearchResult:
        scope(namespace)
        if not isinstance(query, str) or not query.strip():
            raise InvalidInput("query must not be blank")
        options = options or SearchOptions(mode="hybrid" if self.embedder else "lexical")
        if options.representation != "raw" and self.store.extractor_config is None:
            raise ConfigurationError("atom retrieval requires an extracted corpus")
        timings: dict[str, float] = {}
        started = perf_counter()
        plan = await self._plan_query(query, query_options, options)
        timings["embedding"] = (perf_counter() - started) * 1000

        stage = perf_counter()
        retrieval = await asyncio.to_thread(self._retrieve, namespace, plan, options)
        timings["retrieval"] = (perf_counter() - stage) * 1000

        policy = role_policy(options, retrieval.assistant_share)
        candidates = retrieval.candidates
        if policy.boost and options.role_first is not None and options.role_boost is not None:
            candidates = boost_role(
                candidates, retrieval.roles(), options.role_first, options.role_boost
            )
        stage = perf_counter()
        candidates = await self._rerank(plan.model_input, candidates)
        timings["reranking"] = (perf_counter() - stage) * 1000
        if policy.partition and options.role_first is not None:
            candidates = role_first(candidates, retrieval.roles(), options.role_first)

        with_position = bool(options.neighbor_messages)
        evidence = [
            to_evidence(
                c,
                retrieval.windows[c.chunk.id],
                retrieval.neighbors[c.chunk.id],
                with_position=with_position,
            )
            for c in candidates
        ]
        overflow = [
            to_evidence(c, retrieval.windows[c.chunk.id], (), with_position=with_position)
            for c in retrieval.overflow
        ]
        stage = perf_counter()
        packed = await asyncio.to_thread(
            pack,
            evidence,
            self.counter,
            budget=options.token_budget,
            limit=options.limit,
            overflow=overflow,
            # Anchors whose neighbor windows would touch share one item.
            merge_reach=1 + 2 * options.neighbor_messages if options.merge_blocks else None,
            order_key=chronological if options.order == "time" else None,
        )
        timings["packing"] = (perf_counter() - stage) * 1000
        trace = self._trace(options, plan, retrieval, candidates, packed, policy)
        # Measured last, so the total includes building the trace.
        timings["total"] = (perf_counter() - started) * 1000
        return SearchResult(
            evidence=packed.evidence,
            context=packed.context,
            sources=tuple(
                retrieval.sources[id]
                for id in dict.fromkeys(
                    id for entry in packed.evidence for id in entry.source_ids()
                )
            ),
            trace=trace.model_copy(update={"timings_ms": timings}),
        )

    async def _plan_query(
        self, query: str, query_options: Sequence[str] | None, options: SearchOptions
    ) -> QueryPlan:
        full_query = query + ("\n" + "\n".join(query_options) if query_options else "")
        unbounded = query if options.model_query_policy == "question" else full_query
        bounded = model_query(unbounded, counter=self.counter)
        vector = None
        if options.mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise ConfigurationError("dense/hybrid retrieval requires the indexed embedder")
            vector = await self.embedder.embed_query(bounded)
        return QueryPlan(
            text=query,
            options=query_options,
            model_input=bounded,
            original_tokens=self.counter.count(unbounded),
            tokens=self.counter.count(bounded),
            truncated=bounded != unbounded,
            vector=vector,
        )

    def _retrieve(self, namespace: str, plan: QueryPlan, options: SearchOptions) -> Retrieval:
        """Every store read of one search; blocking, so it runs on a worker thread.

        One read transaction, holding the store's lock, covers all channels and the
        materialized sources, so concurrent writes cannot mix versions in the evidence.
        """
        with self.store.transaction(read=True):
            candidate_limit = options.candidate_limit
            chunk_density = None
            if options.deep_candidate_limit is not None:
                chunk_density = self.store.chunk_density(namespace)
                if chunk_density > options.deep_history_density:
                    candidate_limit = options.deep_candidate_limit
            retrieval = Retrieval(
                candidate_limit=candidate_limit,
                chunk_density=chunk_density,
                assistant_share=(
                    self.store.assistant_share(namespace)
                    if options.role_first is not None
                    else None
                ),
            )
            atom_ranks = self._read_channels(namespace, plan, options, retrieval)
            retrieval.candidates = fuse(
                retrieval.channels,
                weights=channel_weights(options),
                k=options.rrf_k,
                ranks=atom_ranks,
            )
            if options.chunks_per_source is not None:
                self._cap_per_source(namespace, options.chunks_per_source, retrieval)
            for candidate in retrieval.overflow:
                retrieval.windows[candidate.chunk.id] = self._window(
                    namespace, candidate.chunk, options, retrieval.sources
                )
            for candidate in retrieval.candidates:
                window = self._window(namespace, candidate.chunk, options, retrieval.sources)
                retrieval.windows[candidate.chunk.id] = window
                retrieval.neighbors[candidate.chunk.id] = self.store.conversation_context(
                    namespace,
                    window.source,
                    neighbors=options.neighbor_messages,
                    max_chars=options.neighbor_chars,
                    source_cache=retrieval.sources,
                )
        return retrieval

    def _read_channels(
        self, namespace: str, plan: QueryPlan, options: SearchOptions, retrieval: Retrieval
    ) -> dict[str, dict[str, int]]:
        """Fill ``retrieval.channels``; return each atom channel's rank by chunk ID."""
        limit = retrieval.candidate_limit
        lexical = options.mode in ("lexical", "hybrid")
        if options.representation != "atoms":
            if lexical:
                retrieval.channels["lexical"] = self.store.lexical(
                    namespace, plan.text, limit, options=plan.options
                )
            if plan.vector is not None:
                retrieval.channels["dense"] = self.store.dense(namespace, plan.vector, limit)
        if options.representation == "raw":
            return {}
        atom_channels = {}
        if lexical:
            atom_channels["atom_lexical"] = self.store.atoms.lexical(
                namespace, plan.text, limit, options=plan.options
            )
        if plan.vector is not None:
            assert self.store.embedding_spec is not None
            atom_channels["atom_dense"] = self.store.atoms.dense(
                namespace,
                vector_blob(plan.vector, self.store.embedding_spec.dimensions),
                limit,
            )
        # An atom ranks the chunks its citations fall in; each chunk keeps the rank
        # and score of the best atom that reached it.
        atom_ranks: dict[str, dict[str, int]] = {}
        for channel, hits in atom_channels.items():
            expanded: dict[str, tuple[Chunk, float]] = {}
            ranks = atom_ranks[channel] = {}
            for rank, (atom, score) in enumerate(hits, 1):
                retrieval.atom_hits.append(
                    AtomHit(atom=atom, channel=channel, rank=rank, score=score)
                )
                for chunk in self.store.atoms.chunks(namespace, atom):
                    expanded.setdefault(chunk.id, (chunk, score))
                    ranks.setdefault(chunk.id, rank)
            retrieval.channels[channel] = list(expanded.values())
        return atom_ranks

    def _cap_per_source(self, namespace: str, per_source: int, retrieval: Retrieval) -> None:
        candidates, overflow = cap_per_source(retrieval.candidates, per_source)
        # Complete each anchored message: the chunks of a long record that no channel
        # retrieved follow its anchor, placed only into unused budget, so a trajectory
        # or document is served whole when the budget allows.
        known = {c.chunk.id for c in candidates} | {c.chunk.id for c in overflow}
        anchored = list(dict.fromkeys(c.chunk.source_id for c in candidates))
        for chunks in self.store.source_chunks(namespace, anchored).values():
            for chunk in chunks:
                if chunk.id not in known:
                    known.add(chunk.id)
                    overflow.append(Candidate(chunk=chunk, fusion_score=0.0, channels={}))
        retrieval.candidates, retrieval.overflow = candidates, overflow

    def _window(
        self, namespace: str, chunk: Chunk, options: SearchOptions, sources: dict[str, Source]
    ) -> Window:
        source = sources.get(chunk.source_id)
        if source is None:
            source = self.store.get_source(namespace, chunk.source_id)
            if source is None:
                raise ValueError("retrieved chunk has no source")
            sources[chunk.source_id] = source
        return self.store.window(namespace, chunk, options.neighbor_chunks, source=source)

    async def _rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if self.reranker is None or not candidates:
            return candidates
        scores = list(await self.reranker.score(query, [item.chunk.text for item in candidates]))
        if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
            raise ValueError("reranker must return one finite score per candidate")
        return sorted(
            (
                item.model_copy(update={"rerank_score": float(score)})
                for item, score in zip(candidates, scores, strict=True)
            ),
            key=by_rerank,
        )

    def _trace(
        self,
        options: SearchOptions,
        plan: QueryPlan,
        retrieval: Retrieval,
        candidates: Sequence[Candidate],
        packed: Packed,
        policy: RolePolicy,
    ) -> SearchTrace:
        neighbors = retrieval.neighbors
        requested_context = {item.id: item for values in neighbors.values() for item in values}
        packed_anchors = [member.id for entry in packed.evidence for member in entry.members()]
        # Only a packed anchor's neighbors can be omitted by budget or duplicate
        # suppression; neighbors of unpacked candidates were never in play.
        anchor_context = {item.id for id in packed_anchors for item in neighbors.get(id, ())}
        included_context = {
            item.id
            for entry in packed.evidence
            for member in entry.members()
            for item in member.neighbors
        }
        return SearchTrace(
            mode=options.mode,
            candidates=tuple(candidates),
            atom_hits=tuple(hit.model_dump(mode="json") for hit in retrieval.atom_hits),
            context_candidates=tuple(requested_context.values()),
            omitted_context_ids=tuple(
                id
                for id in requested_context
                if id in anchor_context and id not in included_context
            ),
            channel_counts={channel: len(items) for channel, items in retrieval.channels.items()},
            packed_tokens=packed.tokens,
            evidence_tokens=sum(self.counter.count(entry.render()) for entry in packed.evidence),
            tokenizer=self.counter.signature,
            omitted_ids=packed.omitted_ids,
            timings_ms={},
            candidate_limit=retrieval.candidate_limit,
            chunk_density=retrieval.chunk_density,
            original_model_query_tokens=plan.original_tokens,
            model_query_tokens=plan.tokens,
            model_query_truncated=plan.truncated,
            assistant_share=retrieval.assistant_share,
            role_first_applied=policy.partition,
            role_boost_applied=policy.boost,
        )

    def source(self, namespace: str, source_id: str) -> Source | None:
        return self.store.get_source(namespace, source_id)

    def sources(self, namespace: str, source_ids: Sequence[str]) -> dict[str, Source]:
        """The stored sources among ``source_ids``, from one snapshot."""
        return self.store.get_sources(namespace, source_ids)

    def forget(self, namespace: str, source_id: str) -> bool:
        return self.store.forget(namespace, source_id)
