"""Source-first ingestion and a single explicit retrieval pipeline."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .core.models import (
    Atom,
    AtomHit,
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
from .core.protocols import Embedder, Extractor, Reranker, TokenCounter
from .errors import ConfigurationError, InvalidInput, ProviderError
from .ingest.atoms import extract_atoms
from .ingest.chunking import chunk_source
from .retrieval.packing import Packed, TiktokenCounter, pack
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
from .storage.store import IngestBatch, SQLiteEvidenceStore, Window, now
from .storage.vectors import vector_blob


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


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Everything one search reads from a single store snapshot."""

    candidate_limit: int
    chunk_density: float | None
    assistant_share: float | None
    channels: dict[str, list[tuple[Chunk, float]]]
    atom_hits: list[AtomHit]
    # Fused anchors, and chunks only placed into budget the anchors leave unused.
    candidates: list[Candidate]
    overflow: list[Candidate]
    windows: dict[str, Window]
    neighbors: dict[str, tuple[ContextSpan, ...]]
    sources: dict[str, Source]

    def roles(self) -> dict[str, str]:
        """Role of each retrieved chunk's source, by chunk ID."""
        return {id: window.source.role for id, window in self.windows.items()}


@dataclass(slots=True)
class Stopwatch:
    """Milliseconds each stage of one search took, and the total."""

    started: float = field(default_factory=perf_counter)
    ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        yield
        self.ms[name] = (perf_counter() - started) * 1000

    def stop(self) -> dict[str, float]:
        self.ms["total"] = (perf_counter() - self.started) * 1000
        return self.ms


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


def _sources(
    namespace: str, messages: Sequence[Message], request_id: str, session_id: str
) -> list[Source]:
    """One source per message, in order; a message without an ID gets a stable one."""
    ingested_at = now()
    return [
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


class EvidenceEngine:
    """Source-first memory over one SQLite file, with every namespace passed per call.

    ``mnemotree.Memory`` binds one namespace over an engine. Configure an embedder for dense/hybrid retrieval; otherwise the default is
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
        if not 1 <= extraction_batch_chars <= 100000:
            raise ConfigurationError("extraction_batch_chars must be between 1 and 100000")
        self.embedder = embedder
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

    def __enter__(self) -> EvidenceEngine:
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
        sources = _sources(namespace, messages, request_id, session_id)
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
            IngestBatch(
                namespace=namespace,
                request_id=request_id,
                session_id=session_id,
                body_hash=body_hash,
                sources=sources,
                chunks=chunks,
                vectors=vectors,
                atoms=atoms,
                atom_vectors=atom_vectors,
            ),
        )

    async def _embed_and_extract(
        self, sources: Sequence[Source], chunks: Sequence[Chunk]
    ) -> tuple[Sequence[Sequence[float]] | None, list[Atom]]:
        """Embed the chunks and extract atoms, concurrently when both providers are set."""
        embedding: asyncio.Task[Sequence[Sequence[float]]] | None = None
        extraction: asyncio.Task[list[Atom]] | None = None
        try:
            async with asyncio.TaskGroup() as group:
                if self.embedder is not None:
                    texts = [chunk.text for chunk in chunks]
                    embedding = group.create_task(self.embedder.embed_documents(texts))
                if self.extractor is not None:
                    extraction = group.create_task(self._extract(self.extractor, sources))
        except ExceptionGroup as failures:
            # Raise the provider's own exception: a group would hide it from callers
            # that catch its type. A failure cancels the other task, so it is the first.
            error = failures.exceptions[0]
        else:
            vectors = embedding.result() if embedding is not None else None
            return vectors, extraction.result() if extraction is not None else []
        # Raised outside the handler so the error keeps its own cause and context.
        raise error

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
        options = options or self.default_options()
        if options.representation != "raw" and self.store.extractor_config is None:
            raise ConfigurationError("atom retrieval requires an extracted corpus")
        watch = Stopwatch()
        with watch.stage("embedding"):
            plan = await self._plan_query(query, query_options, options)
        with watch.stage("retrieval"):
            retrieval = await asyncio.to_thread(self._retrieve, namespace, plan, options)

        policy = role_policy(options, retrieval.assistant_share)
        candidates = retrieval.candidates
        if policy.boost:
            candidates = boost_role(candidates, retrieval.roles(), policy.role, policy.boost_by)
        with watch.stage("reranking"):
            candidates = await self._rerank(plan.model_input, candidates)
        if policy.partition:
            candidates = role_first(candidates, retrieval.roles(), policy.role)

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
        with watch.stage("packing"):
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
        trace = self._trace(options, plan, retrieval, candidates, packed, policy)
        return SearchResult(
            evidence=packed.evidence,
            context=packed.context,
            sources=tuple(
                retrieval.sources[id]
                for id in dict.fromkeys(
                    id for entry in packed.evidence for id in entry.source_ids()
                )
            ),
            # Stopped last, so the total includes building the trace.
            trace=trace.model_copy(update={"timings_ms": watch.stop()}),
        )

    def default_options(self) -> SearchOptions:
        """Hybrid retrieval when an embedder is configured, lexical otherwise."""
        return SearchOptions(mode="hybrid" if self.embedder else "lexical")

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
            assistant_share = (
                self.store.assistant_share(namespace) if options.role_first is not None else None
            )
            channels = self._chunk_channels(namespace, plan, options, candidate_limit)
            atom_hits, atom_channels, atom_ranks = self._atom_channels(
                namespace, plan, options, candidate_limit
            )
            channels.update(atom_channels)
            candidates = fuse(
                channels, weights=channel_weights(options), k=options.rrf_k, ranks=atom_ranks
            )
            overflow: list[Candidate] = []
            if options.chunks_per_source is not None:
                candidates, overflow = self._cap_per_source(
                    namespace, options.chunks_per_source, candidates
                )
            sources: dict[str, Source] = {}
            windows: dict[str, Window] = {}
            neighbors: dict[str, tuple[ContextSpan, ...]] = {}
            for candidate in overflow:
                windows[candidate.chunk.id] = self._window(
                    namespace, candidate.chunk, options, sources
                )
            for candidate in candidates:
                window = self._window(namespace, candidate.chunk, options, sources)
                windows[candidate.chunk.id] = window
                neighbors[candidate.chunk.id] = self.store.conversation_context(
                    namespace,
                    window.source,
                    neighbors=options.neighbor_messages,
                    max_chars=options.neighbor_chars,
                    source_cache=sources,
                )
        return Retrieval(
            candidate_limit=candidate_limit,
            chunk_density=chunk_density,
            assistant_share=assistant_share,
            channels=channels,
            atom_hits=atom_hits,
            candidates=candidates,
            overflow=overflow,
            windows=windows,
            neighbors=neighbors,
            sources=sources,
        )

    def _chunk_channels(
        self, namespace: str, plan: QueryPlan, options: SearchOptions, limit: int
    ) -> dict[str, list[tuple[Chunk, float]]]:
        """Chunks ranked by each raw channel the options ask for."""
        channels: dict[str, list[tuple[Chunk, float]]] = {}
        if options.representation == "atoms":
            return channels
        if options.mode in ("lexical", "hybrid"):
            channels["lexical"] = self.store.lexical(
                namespace, plan.text, limit, options=plan.options
            )
        if plan.vector is not None:
            channels["dense"] = self.store.dense(namespace, plan.vector, limit)
        return channels

    def _atom_channels(
        self, namespace: str, plan: QueryPlan, options: SearchOptions, limit: int
    ) -> tuple[list[AtomHit], dict[str, list[tuple[Chunk, float]]], dict[str, dict[str, int]]]:
        """Atom hits, the chunks they cite per channel, and each chunk's rank there."""
        hits: list[AtomHit] = []
        channels: dict[str, list[tuple[Chunk, float]]] = {}
        ranks: dict[str, dict[str, int]] = {}
        if options.representation == "raw":
            return hits, channels, ranks
        found = {}
        if options.mode in ("lexical", "hybrid"):
            found["atom_lexical"] = self.store.atoms.lexical(
                namespace, plan.text, limit, options=plan.options
            )
        if plan.vector is not None:
            assert self.store.embedding_spec is not None
            found["atom_dense"] = self.store.atoms.dense(
                namespace,
                vector_blob(plan.vector, self.store.embedding_spec.dimensions),
                limit,
            )
        # An atom ranks the chunks its citations fall in; each chunk keeps the rank
        # and score of the best atom that reached it.
        for channel, atoms in found.items():
            expanded: dict[str, tuple[Chunk, float]] = {}
            channel_ranks = ranks[channel] = {}
            for rank, (atom, score) in enumerate(atoms, 1):
                hits.append(AtomHit(atom=atom, channel=channel, rank=rank, score=score))
                for chunk in self.store.atoms.chunks(namespace, atom):
                    expanded.setdefault(chunk.id, (chunk, score))
                    channel_ranks.setdefault(chunk.id, rank)
            channels[channel] = list(expanded.values())
        return hits, channels, ranks

    def _cap_per_source(
        self, namespace: str, per_source: int, candidates: list[Candidate]
    ) -> tuple[list[Candidate], list[Candidate]]:
        """The candidates kept under the cap, and the overflow that may fill spare budget."""
        candidates, overflow = cap_per_source(candidates, per_source)
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
        return candidates, overflow

    def _window(
        self, namespace: str, chunk: Chunk, options: SearchOptions, sources: dict[str, Source]
    ) -> Window:
        source = sources.get(chunk.source_id)
        if source is None:
            source = self.store.get_source(namespace, chunk.source_id)
            if source is None:
                # One read transaction covers the chunk and its source, so this
                # means the database is inconsistent, not that the input is bad.
                raise RuntimeError("retrieved chunk has no source")
            sources[chunk.source_id] = source
        return self.store.window(namespace, chunk, options.neighbor_chunks, source=source)

    async def _rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if self.reranker is None or not candidates:
            return candidates
        scores = list(await self.reranker.score(query, [item.chunk.text for item in candidates]))
        if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
            raise ProviderError("reranker must return one finite score per candidate")
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
