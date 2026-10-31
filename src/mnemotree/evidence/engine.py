"""Source-first ingestion and a single explicit retrieval pipeline."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

from ..errors import ConfigurationError, InvalidInput
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
    identifier,
    scope,
)
from .packing import TiktokenCounter, pack
from .protocols import Embedder, Reranker, TokenCounter
from .store import SQLiteEvidenceStore, now, vector_blob

# Cap the final model input with the configured tokenizer. The service tokenizer
# is a proxy for the provider's tokenizer, so keep margin below its 8,192 limit.
QUERY_EMBED_TOKENS = 6000
QUERY_EMBED_CHARS = 6000


def chronological(item: Evidence) -> tuple[int, float, int]:
    """Earliest member timestamp, then session order; untimed items follow timed ones."""
    members = item.members()
    stamps = [member.timestamp.timestamp() for member in members if member.timestamp is not None]
    seq = min((member.seq for member in members if member.seq is not None), default=0)
    return (0, min(stamps), seq) if stamps else (1, 0.0, seq)


def model_query(
    query: str, limit: int = QUERY_EMBED_CHARS, counter: TokenCounter | None = None
) -> str:
    """Preserve small inputs; otherwise retain endpoints within the final input cap."""
    separator = "\n...\n"

    def endpoints(chars: int) -> str:
        if chars < len(separator):
            return query[:chars]
        retained = chars - len(separator)
        head = (retained + 1) // 2
        tail = retained // 2
        return query[:head] + separator + (query[-tail:] if tail else "")

    if counter is None:
        return query if len(query) <= limit else endpoints(max(0, limit))
    if counter.count(query) <= QUERY_EMBED_TOKENS:
        return query
    # Count every candidate, including its separator. Character/token ratios can
    # differ radically between the middle and endpoints, even in short inputs.
    low, high = 0, len(query) - 1
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = endpoints(middle)
        if counter.count(candidate) <= QUERY_EMBED_TOKENS:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


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


def cap_per_source(
    candidates: Sequence[Candidate], per_source: int
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into the best `per_source` of each source message and the rest.

    Both lists keep the given order. The rest is not discarded: the packer adds it
    once anchors and their neighbors have been placed, so a history that is one long
    document is not served as a single chunk of it.
    """
    seen: dict[str, int] = {}
    kept: list[Candidate] = []
    overflow: list[Candidate] = []
    for candidate in candidates:
        count = seen.get(candidate.chunk.source_id, 0)
        (kept if count < per_source else overflow).append(candidate)
        seen[candidate.chunk.source_id] = count + 1
    return kept, overflow


def role_first(
    candidates: Sequence[Candidate], roles: Mapping[str, str], role: str
) -> list[Candidate]:
    """Stable partition: candidates whose source has `role` first, then the rest."""
    first = [c for c in candidates if roles.get(c.chunk.id) == role]
    rest = [c for c in candidates if roles.get(c.chunk.id) != role]
    return first + rest


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
        started = perf_counter()
        full_query = query + ("\n" + "\n".join(query_options) if query_options else "")
        unbounded_model_query = query if options.model_query_policy == "question" else full_query
        original_model_query_tokens = self.counter.count(unbounded_model_query)
        bounded_query = model_query(unbounded_model_query, counter=self.counter)
        model_query_tokens = self.counter.count(bounded_query)
        query_vector = None
        if options.mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise ConfigurationError("dense/hybrid retrieval requires the indexed embedder")
            query_vector = await self.embedder.embed_query(bounded_query)
        embedding_ms = (perf_counter() - started) * 1000
        retrieval_start = perf_counter()
        channels: dict[str, list[tuple[Chunk, float]]] = {}
        windows: dict[str, tuple[Source, int, int]] = {}
        neighbors: dict[str, tuple[ContextSpan, ...]] = {}
        atom_hits: list[AtomHit] = []
        atom_ranks: dict[str, dict[str, int]] = {}

        def retrieve() -> tuple[float | None, int, float | None, list[Candidate], list[Candidate]]:
            # Every store read of one search; blocking, so it runs on a worker thread.
            # The store's lock is held for the whole transaction.
            # A single snapshot covers all channels and source materialization.
            # Concurrent writes cannot mix different versions in the returned evidence.
            with self.store.transaction(read=True):
                source_cache: dict[str, Source] = {}

                def window(chunk: Chunk) -> tuple[Source, int, int]:
                    source = source_cache.get(chunk.source_id)
                    if source is None:
                        source = self.store.get_source(namespace, chunk.source_id)
                        if source is None:
                            raise ValueError("retrieved chunk has no source")
                        source_cache[chunk.source_id] = source
                    return self.store.window(
                        namespace, chunk, options.neighbor_chunks, source=source
                    )

                assistant_share = (
                    self.store.assistant_share(namespace)
                    if options.role_first is not None
                    else None
                )
                candidate_limit = options.candidate_limit
                chunk_density = None
                if options.deep_candidate_limit is not None:
                    chunk_density = self.store.chunk_density(namespace)
                    if chunk_density > options.deep_history_density:
                        candidate_limit = options.deep_candidate_limit
                if options.representation != "atoms" and options.mode in ("lexical", "hybrid"):
                    channels["lexical"] = self.store.lexical(
                        namespace, query, candidate_limit, options=query_options
                    )
                if options.representation != "atoms" and query_vector is not None:
                    channels["dense"] = self.store.dense(namespace, query_vector, candidate_limit)
                if options.representation != "raw":
                    atom_channels = {}
                    if options.mode in ("lexical", "hybrid"):
                        atom_channels["atom_lexical"] = self.store.atoms.lexical(
                            namespace, query, candidate_limit, options=query_options
                        )
                    if query_vector is not None:
                        assert self.store.embedding_spec is not None
                        atom_channels["atom_dense"] = self.store.atoms.dense(
                            namespace,
                            vector_blob(query_vector, self.store.embedding_spec.dimensions),
                            candidate_limit,
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
                overflow: list[Candidate] = []
                if options.chunks_per_source is not None:
                    candidates, overflow = cap_per_source(candidates, options.chunks_per_source)
                    # Complete each anchored message: the chunks of a long record that no
                    # channel retrieved follow its anchor, placed only into unused budget, so
                    # a trajectory or document is served whole when the budget allows.
                    known = {c.chunk.id for c in candidates} | {c.chunk.id for c in overflow}
                    anchored = list(dict.fromkeys(c.chunk.source_id for c in candidates))
                    for chunks in self.store.source_chunks(namespace, anchored).values():
                        for chunk in chunks:
                            if chunk.id not in known:
                                known.add(chunk.id)
                                overflow.append(
                                    Candidate(chunk=chunk, fusion_score=0.0, channels={})
                                )
                for candidate in overflow:
                    windows[candidate.chunk.id] = window(candidate.chunk)
                for candidate in candidates:
                    windows[candidate.chunk.id] = window(candidate.chunk)
                    source = windows[candidate.chunk.id][0]
                    neighbors[candidate.chunk.id] = self.store.conversation_context(
                        namespace,
                        source,
                        neighbors=options.neighbor_messages,
                        max_chars=options.neighbor_chars,
                    )
            return assistant_share, candidate_limit, chunk_density, candidates, overflow

        (
            assistant_share,
            candidate_limit,
            chunk_density,
            candidates,
            overflow,
        ) = await asyncio.to_thread(retrieve)
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        role_policy_enabled = options.role_first is not None and (
            options.role_first_min_assistant_share is None
            or (
                assistant_share is not None
                and assistant_share >= options.role_first_min_assistant_share
            )
        )
        role_boost_applied = role_policy_enabled and options.role_boost is not None
        role_first_applied = role_policy_enabled and options.role_boost is None
        if role_boost_applied and options.role_first is not None and options.role_boost is not None:
            candidates = sorted(
                (
                    item.model_copy(
                        update={
                            "fusion_score": item.fusion_score
                            + (
                                options.role_boost
                                if windows[item.chunk.id][0].role == options.role_first
                                else 0.0
                            )
                        }
                    )
                    for item in candidates
                ),
                key=lambda item: (-item.fusion_score, item.chunk.id),
            )
        rerank_start = perf_counter()
        if self.reranker and candidates:
            scores = list(
                await self.reranker.score(
                    bounded_query,
                    [item.chunk.text for item in candidates],
                )
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
        if role_first_applied and options.role_first is not None:
            candidates = role_first(
                candidates, {cid: windows[cid][0].role for cid in windows}, options.role_first
            )

        def to_evidence(candidate: Candidate, context: tuple[ContextSpan, ...]) -> Evidence:
            source, start, end = windows[candidate.chunk.id]
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
                request_id=source.request_id if options.neighbor_messages else None,
                position=source.position if options.neighbor_messages else None,
                seq=source.session_seq,
                neighbors=context,
            )

        evidence = [to_evidence(c, neighbors[c.chunk.id]) for c in candidates]
        packing_start = perf_counter()
        selected, context, tokens, omitted = await asyncio.to_thread(
            pack,
            evidence,
            self.counter,
            budget=options.token_budget,
            limit=options.limit,
            overflow=[to_evidence(c, ()) for c in overflow],
            # Anchors whose neighbor windows would touch share one item.
            merge_reach=1 + 2 * options.neighbor_messages if options.merge_blocks else None,
        )
        if options.order == "time":
            selected = tuple(sorted(selected, key=chronological))
            context = "\n\n".join(entry.render() for entry in selected)
        requested_context = {item.id: item for values in neighbors.values() for item in values}
        packed_anchors = [member.id for entry in selected for member in entry.members()]
        # Only a packed anchor's neighbors can be omitted by budget or duplicate
        # suppression; neighbors of unpacked candidates were never in play.
        anchor_context = {item.id for id in packed_anchors for item in neighbors.get(id, ())}
        included_context = {
            item.id for entry in selected for member in entry.members() for item in member.neighbors
        }
        return SearchResult(
            evidence=selected,
            context=context,
            trace=SearchTrace(
                mode=options.mode,
                candidates=tuple(candidates),
                atom_hits=tuple(hit.model_dump(mode="json") for hit in atom_hits),
                context_candidates=tuple(requested_context.values()),
                omitted_context_ids=tuple(
                    id
                    for id in requested_context
                    if id in anchor_context and id not in included_context
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
                candidate_limit=candidate_limit,
                chunk_density=chunk_density,
                original_model_query_tokens=original_model_query_tokens,
                model_query_tokens=model_query_tokens,
                model_query_truncated=bounded_query != unbounded_model_query,
                assistant_share=assistant_share,
                role_first_applied=role_first_applied,
                role_boost_applied=role_boost_applied,
            ),
        )

    def source(self, namespace: str, source_id: str) -> Source | None:
        return self.store.get_source(namespace, source_id)

    def forget(self, namespace: str, source_id: str) -> bool:
        return self.store.forget(namespace, source_id)
