from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..analysis.keywords import KeywordExtractor
from ..ner.base import BaseNER
from ..rerankers import BaseReranker
from ..store.protocols import (
    MemoryCRUDStore,
    SupportsEntityQuery,
    SupportsKnowledgeGraph,
    SupportsMetadataUpdate,
    SupportsStructuredQuery,
    SupportsVectorSearch,
)
from ._internal.indexing import IndexManager
from ._internal.scoring import MemoryScorer, SignalRanker, cosine_similarity
from .models import LinkType, MemoryItem
from .ppr import PPRConfig, build_adjacency_from_links, personalized_pagerank
from .protocols import EmbeddingModel
from .query import MemoryQuery, MemoryQueryBuilder
from .scoring import MemoryScoring

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MAGMA four-graph dimension classification
# ---------------------------------------------------------------------------

SEMANTIC_LINK_TYPES: frozenset[LinkType] = frozenset({
    LinkType.SUPPORTS,
    LinkType.CONTRADICTS,
    LinkType.ELABORATES,
    LinkType.REFERENCES,
    LinkType.SIMILAR_TO,
    LinkType.EXEMPLIFIES,
    LinkType.GENERALIZES,
    LinkType.DERIVES_FROM,
    LinkType.PART_OF,
    LinkType.UPDATES,
    LinkType.SUPERSEDES,
})

TEMPORAL_LINK_TYPES: frozenset[LinkType] = frozenset({
    LinkType.FOLLOWS,
    LinkType.SEQUENCE,
})

CAUSAL_LINK_TYPES: frozenset[LinkType] = frozenset({
    LinkType.CAUSES,
    LinkType.DERIVES_FROM,
})

__all__ = [
    "Retriever",
    "BaseRetriever",
    "VectorEntityRetriever",
    "HybridRetriever",
    "RetrievalStage",
    "FusionStrategy",
    "RetrievalResult",
    "PPRGraphRetriever",
    "SCMRAGRetriever",
    "TypedPathRetriever",
    "rrf_fuse",
]


class RetrievalStage(str, Enum):
    """Stages in the retrieval pipeline."""

    VECTOR = "vector"
    ENTITY = "entity"
    BM25 = "bm25"
    GRAPH = "graph"
    FUSION = "fusion"
    RERANK = "rerank"


class FusionStrategy(str, Enum):
    """Strategies for combining retrieval results."""

    RRF = "reciprocal_rank_fusion"
    WEIGHTED_SUM = "weighted_sum"
    MAX_SCORE = "max_score"
    LEARNED = "learned"


@dataclass
class RetrievalResult:
    """Container for retrieval results with provenance."""

    memory: MemoryItem
    scores: dict[str, float]
    final_score: float
    retrieval_stages: list[RetrievalStage]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@runtime_checkable
class Retriever(Protocol):
    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]: ...


class BaseRetriever:
    def __init__(
        self,
        store: MemoryCRUDStore,
        scoring_system: MemoryScoring,
        ner: BaseNER | None,
        keyword_extractor: KeywordExtractor | None,
        embedder: EmbeddingModel | None,
        index_manager: IndexManager | None = None,
        hyde_embedder: Any = None,
    ) -> None:
        self.store = store
        self.scoring_system = scoring_system
        self.memory_scorer = MemoryScorer(scoring_system)
        self.signal_ranker = SignalRanker()
        self.ner = ner
        self.keyword_extractor = keyword_extractor
        self.embedder = embedder
        self.index_manager = index_manager
        self.hyde_embedder = hyde_embedder  # HyDEEmbedder | None

    async def _get_embedding(self, text: str) -> list[float]:
        if self.hyde_embedder is not None:
            return await self.hyde_embedder.aembed_query(text)
        if not self.embedder:
            raise RuntimeError("Embedder not configured.")
        return await self.embedder.aembed_query(text)

    def _dedupe_memories(self, memories: list[MemoryItem]) -> list[MemoryItem]:
        seen: set[str] = set()
        deduped: list[MemoryItem] = []
        for memory in memories:
            if memory.memory_id in seen:
                continue
            seen.add(memory.memory_id)
            deduped.append(memory)
        return deduped

    async def _update_access(self, memories: list[MemoryItem]) -> None:
        if not isinstance(self.store, SupportsMetadataUpdate):
            return

        update_tasks = []
        for memory in memories:
            memory.update_access()
            update_tasks.append(
                self.store.update_memory_metadata(
                    memory.memory_id,
                    {
                        "last_accessed": memory.last_accessed,
                        "access_count": memory.access_count,
                        "access_history": memory.access_history,
                    },
                )
            )
        if update_tasks:
            await asyncio.gather(*update_tasks)

    async def _query_store(
        self,
        query: MemoryQuery | MemoryQueryBuilder,
    ) -> tuple[list[MemoryItem], list[float] | None]:
        if not isinstance(self.store, SupportsStructuredQuery):
            raise NotImplementedError("This store does not support query_memories().")

        if isinstance(query, MemoryQuery):
            if query.vector is None and not query.filters and not query.relationships:
                raise ValueError("Invalid MemoryQuery")
            return await self.store.query_memories(query), query.vector
        if isinstance(query, MemoryQueryBuilder):
            built = await query.build()
            return await self.store.query_memories(built), built.vector
        raise ValueError("Invalid query type")

    async def _retrieve_vector_candidates(
        self, query: str, limit: int | None
    ) -> tuple[list[MemoryItem], list[float]]:
        query_embedding = await self._get_embedding(query)
        if not isinstance(self.store, SupportsVectorSearch):
            raise NotImplementedError("This store does not support vector similarity search.")

        memories = await self.store.get_similar_memories(
            query=query,
            query_embedding=query_embedding,
            top_k=limit or 10,
        )
        return memories, query_embedding

    async def _retrieve_entity_candidates(self, query: str) -> list[MemoryItem]:
        if not self.ner:
            return []
        if not isinstance(self.store, SupportsEntityQuery):
            return []
        ner_result = await self.ner.extract_entities(query)
        if not ner_result.entities:
            return []
        entity_names = list(ner_result.entities.keys())
        return await self.store.query_by_entities(entity_names)


def rrf_fuse(
    *,
    stage_candidates: Mapping[_StageT, Sequence[MemoryItem]],
    weights: Mapping[_StageT, float] | None = None,
    rrf_k: int = 60,
) -> tuple[list[MemoryItem], dict[str, float], dict[_StageT, dict[str, float]]]:
    scores: dict[str, float] = {}
    stage_scores: dict[_StageT, dict[str, float]] = {}
    memory_by_id: dict[str, MemoryItem] = {}

    for stage, candidates in stage_candidates.items():
        weight = 1.0 if weights is None else weights.get(stage, 0.0)
        if weight <= 0:
            continue

        per_stage = stage_scores.setdefault(stage, {})
        for rank, memory in enumerate(candidates, start=1):
            memory_by_id[memory.memory_id] = memory
            contribution = weight / (rrf_k + rank)
            scores[memory.memory_id] = scores.get(memory.memory_id, 0.0) + contribution
            per_stage[memory.memory_id] = contribution

    fused = list(memory_by_id.values())
    fused.sort(key=lambda memory: scores.get(memory.memory_id, 0.0), reverse=True)
    return fused, scores, stage_scores


_StageT = TypeVar("_StageT", bound=Hashable)


class VectorEntityRetriever(BaseRetriever):
    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
        started = time.perf_counter()
        query_keywords: list[str] = []
        entity_memory_ids: set[str] = set()
        if isinstance(query, str):
            logger.debug("vector_entity recall start limit=%s scoring=%s", limit, scoring)
            (
                memories,
                query_embedding,
                query_keywords,
                entity_memory_ids,
            ) = await self._recall_from_text(query, limit)
        else:
            memories, query_embedding = await self._query_store(query)

        if scoring:
            memories = self.memory_scorer.rank(
                memories,
                query_embedding,
            )

        if isinstance(query, str) and query_embedding and (query_keywords or entity_memory_ids):
            memories = self.signal_ranker.rank(
                memories,
                query_embedding,
                extra_signals={
                    "query_keywords": query_keywords,
                    "entity_memory_ids": entity_memory_ids,
                },
            )

        if limit is not None:
            memories = memories[:limit]

        if update_access and memories:
            await self._update_access(memories)

        logger.debug(
            "vector_entity recall done returned=%d duration_ms=%.2f",
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )

        return memories

    async def _recall_from_text(
        self, query: str, limit: int | None
    ) -> tuple[list[MemoryItem], list[float] | None, list[str], set[str]]:
        vector_task = asyncio.create_task(self._retrieve_vector_candidates(query, limit))
        entity_task = asyncio.create_task(self._retrieve_entity_candidates(query))
        keywords_task = (
            asyncio.create_task(self.keyword_extractor.extract(query))
            if self.keyword_extractor
            else None
        )

        (vector_memories, query_embedding), entity_memories = await asyncio.gather(
            vector_task,
            entity_task,
        )
        query_keywords = await keywords_task if keywords_task else []
        entity_memories, entity_memory_ids = self._filter_entity_candidates(
            entity_memories, query_embedding
        )
        memories = self._dedupe_memories(vector_memories + entity_memories)

        logger.debug(
            "vector_entity recall staged vector=%d entity=%d keywords=%d deduped=%d",
            len(vector_memories),
            len(entity_memories),
            len(query_keywords),
            len(memories),
        )
        return memories, query_embedding, query_keywords, entity_memory_ids

    @staticmethod
    def _filter_entity_candidates(
        entity_memories: list[MemoryItem],
        query_embedding: list[float] | None,
        *,
        min_entity_similarity: float = 0.15,
    ) -> tuple[list[MemoryItem], set[str]]:
        if not entity_memories or not query_embedding:
            return entity_memories, set()

        scored_entities: list[tuple[float, MemoryItem]] = []
        for memory in entity_memories:
            similarity = cosine_similarity(memory.embedding, query_embedding)
            scored_entities.append((similarity, memory))

        filtered = [
            memory for similarity, memory in scored_entities if similarity >= min_entity_similarity
        ]
        entity_memory_ids = {memory.memory_id for memory in filtered}
        return filtered, entity_memory_ids


class HybridRetriever(BaseRetriever):
    """Unified multi-stage hybrid retrieval pipeline.

    Combines vector similarity, entity matching, BM25, and optional graph
    traversal with configurable fusion scoring and cross-encoder reranking.

    Can be used in two modes:
    - **Integrated** (via ``recall()``): collects candidates internally using
      store/embedder from BaseRetriever.
    - **Standalone** (via ``retrieve()``): accepts pre-scored candidate lists
      from external sources.
    """

    def __init__(
        self,
        *args: Any,
        vector_weight: float = 0.50,
        entity_weight: float = 0.15,
        bm25_weight: float = 0.35,
        graph_weight: float = 0.0,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        rrf_k: int = 60,
        enable_rrf_signal_rerank: bool = False,
        reranker: BaseReranker | None = None,
        rerank_candidates: int = 50,
        # Allow standalone mode (retrieve() only) without store/embedder.
        store: MemoryCRUDStore | None = None,
        embedder: EmbeddingModel | None = None,
        ner: BaseNER | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        scoring_system: MemoryScoring | None = None,
        index_manager: IndexManager | None = None,
        memory_scoring: MemoryScoring | None = None,
        hyde_embedder: Any = None,
        **kwargs: Any,
    ) -> None:
        # Support both positional (BaseRetriever style) and keyword-only args.
        # When no positional args and no store/embedder, run in standalone mode.
        resolved_scoring = scoring_system or memory_scoring or MemoryScoring()
        if args or store is not None:
            if store is not None and not args:
                super().__init__(
                    store=store,
                    scoring_system=resolved_scoring,
                    ner=ner,
                    keyword_extractor=keyword_extractor,
                    embedder=embedder,
                    index_manager=index_manager,
                    hyde_embedder=hyde_embedder,
                    **kwargs,
                )
            else:
                super().__init__(*args, **kwargs)
            self._standalone = False
        else:
            # Standalone mode — retrieve() only, no store/embedder required.
            self._standalone = True
            self.store = None  # type: ignore[assignment]
            self.scoring_system = resolved_scoring
            self.memory_scorer = MemoryScorer(resolved_scoring)
            self.signal_ranker = SignalRanker()
            self.ner = ner
            self.keyword_extractor = keyword_extractor
            self.embedder = embedder
            self.index_manager = index_manager
            self.hyde_embedder = hyde_embedder
        self.weights = {
            RetrievalStage.VECTOR: vector_weight,
            RetrievalStage.ENTITY: entity_weight,
            RetrievalStage.BM25: bm25_weight,
            RetrievalStage.GRAPH: graph_weight,
        }
        self._normalize_weights()
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.enable_rrf_signal_rerank = enable_rrf_signal_rerank
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # recall() — integrated mode (Retriever protocol)
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
        if self._standalone:
            raise RuntimeError(
                "HybridRetriever.recall() requires store and embedder. "
                "Pass store= and embedder= at construction, or use retrieve() with explicit candidates."
            )
        started = time.perf_counter()
        if not isinstance(query, str):
            return await self._recall_structured(
                query=query,
                limit=limit,
                scoring=scoring,
                update_access=update_access,
                started=started,
            )

        resolved_limit = limit if limit is not None else 10
        logger.debug(
            "hybrid recall start limit=%s resolved_limit=%d scoring=%s rerank=%s",
            limit,
            resolved_limit,
            scoring,
            bool(self.reranker and self.rerank_candidates > 0),
        )
        cache_len = self.index_manager.doc_count if self.index_manager else 0
        candidate_k = min(
            max(50, resolved_limit * 5),
            max(resolved_limit, cache_len or resolved_limit),
        )

        (
            vector_memories,
            entity_memories,
            bm25_memories,
            query_embedding,
            entity_memory_ids,
            keyword_task,
        ) = await self._collect_candidates(query, candidate_k)

        # Collect graph candidates if weight > 0 and store supports it
        stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]] = {
            RetrievalStage.VECTOR: self._score_candidates(vector_memories, query_embedding),
            RetrievalStage.ENTITY: self._score_candidates(entity_memories, query_embedding),
            RetrievalStage.BM25: [
                (m, 1.0 / (i + 1)) for i, m in enumerate(bm25_memories)
            ],
        }

        if (
            self.weights.get(RetrievalStage.GRAPH, 0.0) > 0
            and isinstance(self.store, SupportsKnowledgeGraph)
            and vector_memories
        ):
            graph_candidates = await self._collect_graph_candidates(
                vector_memories, candidate_k, query_embedding
            )
            if graph_candidates:
                stage_candidates[RetrievalStage.GRAPH] = graph_candidates

        # Fuse using configured strategy
        results = self._fuse_candidates(stage_candidates)

        memories = [r.memory for r in results]
        rrf_scores = {r.memory.memory_id: r.final_score for r in results}

        if scoring:
            memories = self.memory_scorer.rank(memories, query_embedding)

        if self.enable_rrf_signal_rerank and query_embedding:
            try:
                query_keywords = await keyword_task if keyword_task else []
            except Exception:
                logger.debug("Keyword extraction failed, proceeding without keywords", exc_info=True)
                query_keywords = []
            memories = self.signal_ranker.rank(
                memories,
                query_embedding,
                extra_signals={
                    "rrf_scores": rrf_scores,
                    "query_keywords": query_keywords,
                    "entity_memory_ids": entity_memory_ids,
                },
            )

        memories = await self._maybe_rerank(query, memories)

        if limit is not None:
            memories = memories[:limit]
        if update_access and memories:
            await self._update_access(memories)

        logger.debug(
            "hybrid recall done returned=%d duration_ms=%.2f",
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return memories

    # ------------------------------------------------------------------
    # retrieve() — standalone mode (explicit candidates)
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        vector_candidates: list[tuple[MemoryItem, float]],
        entity_candidates: list[tuple[MemoryItem, float]],
        graph_candidates: list[tuple[MemoryItem, float]] | None = None,
        bm25_candidates: list[tuple[MemoryItem, float]] | None = None,
        top_k: int = 10,
        apply_reranking: bool = True,
    ) -> list[RetrievalResult]:
        stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]] = {
            RetrievalStage.VECTOR: vector_candidates,
            RetrievalStage.ENTITY: entity_candidates,
        }
        if graph_candidates:
            stage_candidates[RetrievalStage.GRAPH] = graph_candidates
        if bm25_candidates:
            stage_candidates[RetrievalStage.BM25] = bm25_candidates

        fused_results = self._fuse_candidates(stage_candidates)

        if apply_reranking and self.reranker:
            memories = [r.memory for r in fused_results]
            reranked = await self.reranker.rerank(query, memories, top_k=top_k * 2)
            rerank_map = {mem.memory_id: score for mem, score in reranked}
            for result in fused_results:
                if result.memory.memory_id in rerank_map:
                    result.scores[RetrievalStage.RERANK.value] = rerank_map[result.memory.memory_id]
                    result.retrieval_stages.append(RetrievalStage.RERANK)
                    result.final_score = (
                        0.6 * result.final_score + 0.4 * rerank_map[result.memory.memory_id]
                    )
            fused_results.sort(key=lambda x: x.final_score, reverse=True)

        return fused_results[:top_k]

    # ------------------------------------------------------------------
    # Internal: structured query fallback
    # ------------------------------------------------------------------

    async def _recall_structured(
        self,
        *,
        query: MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
        started: float,
    ) -> list[MemoryItem]:
        memories, query_embedding = await self._query_store(query)
        if scoring:
            memories = self.memory_scorer.rank(memories, query_embedding)
        if limit is not None:
            memories = memories[:limit]
        if update_access and memories:
            await self._update_access(memories)
        logger.debug(
            "hybrid recall done (structured) returned=%d duration_ms=%.2f",
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return memories

    # ------------------------------------------------------------------
    # Internal: candidate collection
    # ------------------------------------------------------------------

    async def _collect_candidates(
        self, query: str, candidate_k: int
    ) -> tuple[
        list[MemoryItem],
        list[MemoryItem],
        list[MemoryItem],
        list[float] | None,
        set[str],
        asyncio.Task[list[str]] | None,
    ]:
        vector_task = asyncio.create_task(self._retrieve_vector_candidates(query, candidate_k))
        entity_task = asyncio.create_task(self._retrieve_entity_candidates(query))
        keyword_task = (
            asyncio.create_task(self.keyword_extractor.extract(query))
            if self.enable_rrf_signal_rerank and self.keyword_extractor
            else None
        )

        (vector_memories, query_embedding), entity_memories = await asyncio.gather(
            vector_task, entity_task
        )

        entity_memories, entity_memory_ids = VectorEntityRetriever._filter_entity_candidates(
            entity_memories, query_embedding
        )
        if query_embedding:
            entity_memories = sorted(
                entity_memories,
                key=lambda memory: cosine_similarity(memory.embedding, query_embedding)
                if memory.embedding
                else 0.0,
                reverse=True,
            )

        bm25_memories: list[MemoryItem] = []
        if self.index_manager:
            ranked = self.index_manager.search(query, k=candidate_k)
            for memory_id, _ in ranked:
                cached_memory = self.index_manager.get_memory(memory_id)
                if cached_memory:
                    bm25_memories.append(cached_memory)

        logger.debug(
            "hybrid recall staged vector=%d entity=%d bm25=%d candidate_k=%d",
            len(vector_memories),
            len(entity_memories),
            len(bm25_memories),
            candidate_k,
        )
        return (
            vector_memories,
            entity_memories,
            bm25_memories,
            query_embedding,
            entity_memory_ids,
            keyword_task,
        )

    # ------------------------------------------------------------------
    # Internal: MAGMA four-graph candidate collection
    # ------------------------------------------------------------------

    async def _collect_graph_candidates(
        self,
        seed_memories: list[MemoryItem],
        candidate_k: int,
        query_embedding: list[float] | None,
    ) -> list[tuple[MemoryItem, float]]:
        """Traverse semantic, temporal, causal, and entity graphs from seed memories.

        Uses the MAGMA four-graph pattern: each graph dimension is traversed
        independently, then results are merged with depth-based scoring.
        """
        assert isinstance(self.store, SupportsKnowledgeGraph)
        seed_ids = [m.memory_id for m in seed_memories[:5]]  # Limit seeds

        # Traverse three explicit graph dimensions in parallel
        graph_dims: list[tuple[str, list[LinkType]]] = [
            ("semantic", list(SEMANTIC_LINK_TYPES)),
            ("temporal", list(TEMPORAL_LINK_TYPES)),
            ("causal", list(CAUSAL_LINK_TYPES)),
        ]
        tasks = [
            self.store.traverse_graph(
                sid,
                max_depth=2,
                link_types=link_types,
            )
            for sid in seed_ids
            for _, link_types in graph_dims
        ]

        if not tasks:
            return []

        all_traversals = await asyncio.gather(*tasks, return_exceptions=True)

        # Score by inverse depth
        scored: dict[str, tuple[MemoryItem, float]] = {}
        seed_id_set = set(seed_ids)
        for result in all_traversals:
            if isinstance(result, BaseException):
                logger.debug("graph traversal error: %s", result)
                continue
            for memory, depth, _path in result:
                if memory.memory_id in seed_id_set:
                    continue
                score = 1.0 / (1.0 + depth)
                if memory.memory_id in scored:
                    # Keep higher score
                    existing = scored[memory.memory_id]
                    if score > existing[1]:
                        scored[memory.memory_id] = (memory, score)
                else:
                    scored[memory.memory_id] = (memory, score)

        # Entity graph: find memories sharing entities with seeds
        seed_entities: set[str] = set()
        for m in seed_memories[:5]:
            if m.entities:
                seed_entities.update(m.entities.keys())
        if seed_entities and isinstance(self.store, SupportsEntityQuery):
            try:
                entity_neighbors = await self.store.query_by_entities(
                    list(seed_entities)[:10]
                )
                for mem in entity_neighbors:
                    if mem.memory_id not in seed_id_set and mem.memory_id not in scored:
                        sim = 0.5  # default entity co-occurrence score
                        if query_embedding and mem.embedding:
                            sim = max(0.0, cosine_similarity(mem.embedding, query_embedding))
                        scored[mem.memory_id] = (mem, sim)
            except Exception:
                logger.debug("Entity graph query failed", exc_info=True)

        candidates = list(scored.values())
        candidates.sort(key=lambda x: x[1], reverse=True)
        logger.debug("graph candidates collected=%d", len(candidates))
        return candidates[:candidate_k]

    # ------------------------------------------------------------------
    # Internal: reranking
    # ------------------------------------------------------------------

    async def _maybe_rerank(self, query: str, memories: list[MemoryItem]) -> list[MemoryItem]:
        if not self.reranker or self.rerank_candidates <= 0:
            return memories
        rerank_limit = min(self.rerank_candidates, len(memories))
        if rerank_limit <= 0:
            return memories
        rerank_slice = memories[:rerank_limit]
        try:
            reranked_tuples = await self.reranker.rerank(query, rerank_slice)
        except Exception:
            logger.debug("Reranking failed, returning original order", exc_info=True)
            return memories
        reranked = [memory for memory, _score in reranked_tuples]
        return reranked + memories[rerank_limit:]

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _score_candidates(
        memories: list[MemoryItem],
        query_embedding: list[float] | None,
    ) -> list[tuple[MemoryItem, float]]:
        if not memories:
            return []
        if not query_embedding:
            return [(m, 0.0) for m in memories]
        return [
            (m, max(0.0, cosine_similarity(m.embedding, query_embedding)))
            for m in memories
        ]

    def _fuse_candidates(
        self, stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]]
    ) -> list[RetrievalResult]:
        if self.fusion_strategy == FusionStrategy.RRF:
            return self._reciprocal_rank_fusion(stage_candidates)
        elif self.fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum_fusion(stage_candidates)
        elif self.fusion_strategy == FusionStrategy.MAX_SCORE:
            return self._max_score_fusion(stage_candidates)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _reciprocal_rank_fusion(
        self,
        stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]],
    ) -> list[RetrievalResult]:
        memory_candidates = {
            stage: [memory for memory, _score in candidates]
            for stage, candidates in stage_candidates.items()
        }
        fused, scores, stage_scores = rrf_fuse(
            stage_candidates=memory_candidates,
            weights=self.weights,
            rrf_k=self.rrf_k,
        )
        results: list[RetrievalResult] = []
        for memory in fused:
            mid = memory.memory_id
            result = RetrievalResult(
                memory=memory,
                scores={},
                final_score=scores.get(mid, 0.0),
                retrieval_stages=[],
            )
            for stage, per_stage in stage_scores.items():
                if mid in per_stage:
                    result.scores[stage.value] = per_stage[mid]
                    result.retrieval_stages.append(stage)
            results.append(result)
        return results

    def _weighted_sum_fusion(
        self,
        stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]],
    ) -> list[RetrievalResult]:
        memory_map: dict[str, RetrievalResult] = {}
        for stage, candidates in stage_candidates.items():
            weight = self.weights.get(stage, 0.0)
            if not candidates:
                continue
            scores = [score for _, score in candidates]
            max_score = max(scores) if scores else 1.0
            for memory, score in candidates:
                normalized = score / max_score if max_score > 0 else 0.0
                weighted = weight * normalized
                result = self._ensure_result(memory_map, memory)
                result.scores[stage.value] = weighted
                result.final_score += weighted
                if stage not in result.retrieval_stages:
                    result.retrieval_stages.append(stage)
        results = list(memory_map.values())
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _max_score_fusion(
        self,
        stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]],
    ) -> list[RetrievalResult]:
        memory_map: dict[str, RetrievalResult] = {}
        for stage, candidates in stage_candidates.items():
            for memory, score in candidates:
                result = self._ensure_result(memory_map, memory)
                result.scores[stage.value] = score
                result.final_score = max(result.final_score, score)
                if stage not in result.retrieval_stages:
                    result.retrieval_stages.append(stage)
        results = list(memory_map.values())
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    @staticmethod
    def _ensure_result(
        memory_map: dict[str, RetrievalResult],
        memory: MemoryItem,
    ) -> RetrievalResult:
        if memory.memory_id not in memory_map:
            memory_map[memory.memory_id] = RetrievalResult(
                memory=memory,
                scores={},
                final_score=0.0,
                retrieval_stages=[],
            )
        return memory_map[memory.memory_id]


# ---------------------------------------------------------------------------
# PPRGraphRetriever
# ---------------------------------------------------------------------------


class PPRGraphRetriever(BaseRetriever):
    """Graph-augmented retriever using Personalized PageRank.

    Runs standard hybrid retrieval to find seed memories, then propagates
    activation through the knowledge-graph link structure via PPR to surface
    multi-hop relevant memories.
    """

    def __init__(
        self,
        *args: Any,
        ppr_cfg: PPRConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ppr_cfg = ppr_cfg or PPRConfig()

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
        started = time.perf_counter()
        if not isinstance(query, str):
            # Fall back to vector retrieval for structured queries
            memories, query_embedding = await self._query_store(query)
            if limit is not None:
                memories = memories[:limit]
            if update_access and memories:
                await self._update_access(memories)
            return memories

        resolved_limit = limit if limit is not None else 10
        cfg = self.ppr_cfg

        # 1. Seed retrieval via vector search
        vector_memories, query_embedding = await self._retrieve_vector_candidates(
            query, resolved_limit * 3
        )

        if not vector_memories:
            return []

        # 2. Build seed scores (cosine similarity)
        seed_scores: dict[str, float] = {}
        for memory in vector_memories:
            sim = cosine_similarity(memory.embedding, query_embedding)
            seed_scores[memory.memory_id] = max(0.0, sim)

        # 3. Fetch neighbourhood links (requires SupportsKnowledgeGraph)
        if not isinstance(self.store, SupportsKnowledgeGraph):
            logger.warning("PPRGraphRetriever: store does not support SupportsKnowledgeGraph; falling back to seed results")
            if limit is not None:
                vector_memories = vector_memories[:limit]
            return vector_memories

        link_types: list[LinkType] | None = (
            [LinkType(lt) for lt in cfg.link_types] if cfg.link_types else None
        )
        links = await self.store.get_neighborhood_links(
            list(seed_scores),
            max_depth=cfg.max_depth,
            link_types=link_types,
        )

        # 4. Build adjacency and run PPR
        adjacency = build_adjacency_from_links(links, cfg)
        ppr_scores = personalized_pagerank(seed_scores, adjacency, cfg)

        if not ppr_scores:
            if limit is not None:
                vector_memories = vector_memories[:limit]
            return vector_memories

        # 5. Fetch MemoryItems for top PPR nodes not already in seed results
        seed_ids = {m.memory_id for m in vector_memories}
        extra_ids = [
            mid for mid in ppr_scores if mid not in seed_ids
        ]
        extra_memories: list[MemoryItem] = []
        if extra_ids:
            fetch_tasks = [self.store.get_memory(mid) for mid in extra_ids]
            fetched = await asyncio.gather(*fetch_tasks)
            extra_memories = [m for m in fetched if m is not None]

        all_memories_by_id: dict[str, MemoryItem] = {
            m.memory_id: m for m in vector_memories + extra_memories
        }

        # 6. Blend PPR + cosine scores
        ppr_weight = cfg.ppr_weight
        blended: list[tuple[float, MemoryItem]] = []
        for mid, memory in all_memories_by_id.items():
            ppr_s = ppr_scores.get(mid, 0.0)
            cosine_s = seed_scores.get(mid, 0.0)
            final = ppr_weight * ppr_s + (1.0 - ppr_weight) * cosine_s
            blended.append((final, memory))

        blended.sort(key=lambda t: t[0], reverse=True)
        memories = [m for _, m in blended]

        if scoring:
            memories = self.memory_scorer.rank(memories, query_embedding)

        if limit is not None:
            memories = memories[:limit]

        if update_access and memories:
            await self._update_access(memories)

        logger.debug(
            "ppr recall done seed=%d extra=%d returned=%d duration_ms=%.2f",
            len(vector_memories),
            len(extra_memories),
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return memories


# ---------------------------------------------------------------------------
# SCMRAGRetriever
# ---------------------------------------------------------------------------


class SCMRAGRetriever(BaseRetriever):
    """Self-Corrective Multi-Round Retrieval (SCMRAG-style).

    After an initial retrieval pass, a GapAnalyzer detects missing information
    and issues targeted sub-queries to fill those gaps.  Runs for at most
    `max_correction_rounds` correction rounds.
    """

    def __init__(
        self,
        *args: Any,
        base_retriever: Retriever,
        gap_analyzer: Any,  # GapAnalyzer protocol - avoid circular import
        max_correction_rounds: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_retriever = base_retriever
        self.gap_analyzer = gap_analyzer
        self.max_correction_rounds = max_correction_rounds

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
        started = time.perf_counter()
        if not isinstance(query, str):
            return await self.base_retriever.recall(query, limit, scoring, update_access)

        resolved_limit = limit if limit is not None else 10
        candidate_limit = resolved_limit * 2

        # Initial retrieval
        initial = await self.base_retriever.recall(
            query, candidate_limit, scoring=False, update_access=False
        )
        all_results: dict[str, MemoryItem] = {m.memory_id: m for m in initial}

        # Extract query entities (best-effort; NER may not be configured)
        query_entities: dict[str, str] = {}
        if self.ner:
            try:
                ner_result = await self.ner.extract_entities(query)
                query_entities = ner_result.entities or {}
            except Exception:
                logger.debug("NER extraction failed", exc_info=True)

        # Correction rounds
        for _ in range(self.max_correction_rounds):
            sub_queries = await self.gap_analyzer.detect_gaps(
                query, list(all_results.values()), query_entities
            )
            if not sub_queries:
                break

            extra_tasks = [
                self.base_retriever.recall(sq, resolved_limit, scoring=False, update_access=False)
                for sq in sub_queries
            ]
            extra_batches = await asyncio.gather(*extra_tasks)
            added = 0
            for batch in extra_batches:
                for m in batch:
                    if m.memory_id not in all_results:
                        all_results[m.memory_id] = m
                        added += 1
            if added == 0:
                break

        # Re-rank all candidates
        all_memories = list(all_results.values())
        if scoring:
            query_embedding = await self._get_embedding(query)
            all_memories = self.memory_scorer.rank(all_memories, query_embedding)
        else:
            all_memories.sort(key=lambda m: m.importance, reverse=True)

        if limit is not None:
            all_memories = all_memories[:limit]

        if update_access and all_memories:
            await self._update_access(all_memories)

        logger.debug(
            "scmrag recall done initial=%d total=%d returned=%d duration_ms=%.2f",
            len(initial),
            len(all_results),
            len(all_memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return all_memories


# ---------------------------------------------------------------------------
# TypedPathRetriever
# ---------------------------------------------------------------------------


class TypedPathRetriever(BaseRetriever):
    """Semantic GraphRAG-style retriever using typed-path traversal.

    Seeds are found via vector search, then each seed is used as a start node
    for `traverse_typed_path` with the caller-supplied path_pattern.  Results
    are merged and optionally re-scored via PPR.
    """

    def __init__(
        self,
        *args: Any,
        path_pattern: list[LinkType],
        ppr_cfg: PPRConfig | None = None,
        min_strength: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.path_pattern = path_pattern
        self.ppr_cfg = ppr_cfg
        self.min_strength = min_strength

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
        started = time.perf_counter()
        if not isinstance(query, str):
            return (await self._query_store(query))[0]

        resolved_limit = limit if limit is not None else 10

        if not isinstance(self.store, SupportsKnowledgeGraph):
            logger.warning("TypedPathRetriever: store does not support SupportsKnowledgeGraph; returning empty")
            return []

        # Seed retrieval
        seed_memories, query_embedding = await self._retrieve_vector_candidates(
            query, resolved_limit * 2
        )

        # Typed-path traversal from each seed
        traverse_tasks = [
            self.store.traverse_typed_path(
                m.memory_id,
                self.path_pattern,
                max_results=resolved_limit,
            )
            for m in seed_memories
        ]
        all_traversals = await asyncio.gather(*traverse_tasks)

        collected: dict[str, MemoryItem] = {m.memory_id: m for m in seed_memories}
        for traversal in all_traversals:
            for end_memory, _path_links in traversal:
                collected.setdefault(end_memory.memory_id, end_memory)

        memories = list(collected.values())

        if scoring:
            memories = self.memory_scorer.rank(memories, query_embedding)
        else:
            memories.sort(key=lambda m: m.importance, reverse=True)

        if limit is not None:
            memories = memories[:limit]

        if update_access and memories:
            await self._update_access(memories)

        logger.debug(
            "typed_path recall done seed=%d total=%d returned=%d duration_ms=%.2f",
            len(seed_memories),
            len(collected),
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return memories
