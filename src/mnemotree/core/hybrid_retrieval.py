"""
Multi-stage hybrid retrieval pipeline with fusion scoring and cross-encoder reranking.

This module implements a sophisticated retrieval strategy:
1. Vector similarity search (semantic retrieval)
2. Entity/graph-based retrieval (structured knowledge)
3. Fusion scoring (combining multiple signals)
4. Optional cross-encoder reranking (semantic precision)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.embeddings.embeddings import Embeddings

from ..analysis.keywords import KeywordExtractor
from ..ner.base import BaseNER
from ..rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker
from ..store.protocols import MemoryCRUDStore
from ._internal.indexing import IndexManager
from .models import MemoryItem
from .query import MemoryQuery, MemoryQueryBuilder
from .retrieval import BaseRetriever, VectorEntityRetriever, rrf_fuse
from .scoring import MemoryScoring, cosine_similarity

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalStage",
    "FusionStrategy",
    "RetrievalResult",
    "HybridRetriever",
    "BaseReranker",
    "CrossEncoderReranker",
    "NoOpReranker",
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

    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # Weighted score combination
    MAX_SCORE = "max_score"  # Take maximum score
    LEARNED = "learned"  # ML-based fusion (future)


@dataclass
class RetrievalResult:
    """Container for retrieval results with provenance."""

    memory: MemoryItem
    scores: dict[str, float]  # Score per stage
    final_score: float
    retrieval_stages: list[RetrievalStage]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """
    Multi-stage hybrid retrieval pipeline.

    Combines vector similarity, entity matching, and graph traversal
    with fusion scoring and optional cross-encoder reranking.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        entity_weight: float = 0.3,
        graph_weight: float = 0.2,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        reranker: BaseReranker | None = None,
        memory_scoring: MemoryScoring | None = None,
        *,
        bm25_weight: float = 0.0,
        store: MemoryCRUDStore | None = None,
        embedder: Embeddings | None = None,
        ner: BaseNER | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        index_manager: IndexManager | None = None,
        rrf_k: int = 60,
        enable_rrf_signal_rerank: bool = False,
        rerank_candidates: int = 50,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_weight: Weight for vector similarity scores
            entity_weight: Weight for entity matching scores
            graph_weight: Weight for graph traversal scores
            fusion_strategy: Strategy for combining retrieval signals
            reranker: Optional reranker for final stage
            memory_scoring: Optional memory scoring system
        """
        self.weights = {
            RetrievalStage.VECTOR: vector_weight,
            RetrievalStage.ENTITY: entity_weight,
            RetrievalStage.BM25: bm25_weight,
            RetrievalStage.GRAPH: graph_weight,
        }
        self._normalize_weights()
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker or NoOpReranker()
        self.memory_scoring = memory_scoring
        self.rrf_k = rrf_k
        self.enable_rrf_signal_rerank = enable_rrf_signal_rerank
        self.rerank_candidates = rerank_candidates
        self._backend: BaseRetriever | None = (
            BaseRetriever(
                store=store,
                scoring_system=memory_scoring or MemoryScoring(),
                ner=ner,
                keyword_extractor=keyword_extractor,
                embedder=embedder,
                index_manager=index_manager,
            )
            if store is not None and embedder is not None
            else None
        )

    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

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
        """
        Execute multi-stage hybrid retrieval.

        Args:
            query: Search query
            vector_candidates: Results from vector search with scores
            entity_candidates: Results from entity matching with scores
            graph_candidates: Optional results from graph traversal with scores
            top_k: Number of final results to return
            apply_reranking: Whether to apply cross-encoder reranking

        Returns:
            List of RetrievalResult objects sorted by final score
        """
        # Stage 1-3: Collect candidates from different sources
        stage_candidates = {
            RetrievalStage.VECTOR: vector_candidates,
            RetrievalStage.ENTITY: entity_candidates,
        }

        if graph_candidates:
            stage_candidates[RetrievalStage.GRAPH] = graph_candidates
        if bm25_candidates:
            stage_candidates[RetrievalStage.BM25] = bm25_candidates

        # Stage 4: Fusion - combine scores from different stages
        fused_results = self._fuse_candidates(stage_candidates)

        # Stage 5: Optional reranking
        if apply_reranking and self.reranker:
            memories = [r.memory for r in fused_results]
            reranked = await self.reranker.rerank(query, memories, top_k=top_k * 2)

            # Update results with reranking scores
            rerank_map = {mem.memory_id: score for mem, score in reranked}
            for result in fused_results:
                if result.memory.memory_id in rerank_map:
                    result.scores[RetrievalStage.RERANK] = rerank_map[result.memory.memory_id]
                    result.retrieval_stages.append(RetrievalStage.RERANK)
                    # Combine fusion score with reranking score
                    result.final_score = (
                        0.6 * result.final_score + 0.4 * rerank_map[result.memory.memory_id]
                    )

            # Re-sort by updated scores
            fused_results.sort(key=lambda x: x.final_score, reverse=True)

        return fused_results[:top_k]

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
        """
        Retrieve memories based on a query using the hybrid pipeline.

        This mirrors the core Retriever protocol and requires store/embedder wiring.
        """
        backend = self._require_backend()
        started = time.perf_counter()

        if not isinstance(query, str):
            memories, query_embedding = await backend._query_store(query)
            if scoring:
                memories = backend.memory_scorer.rank(memories, query_embedding)
            if limit is not None:
                memories = memories[:limit]
            if update_access and memories:
                await backend._update_access(memories)
            logger.debug(
                "hybrid recall done (structured) returned=%d duration_ms=%.2f",
                len(memories),
                (time.perf_counter() - started) * 1000.0,
            )
            return memories

        resolved_limit = limit if limit is not None else 10
        cache_len = backend.index_manager.doc_count if backend.index_manager else 0
        candidate_k = min(
            max(50, resolved_limit * 5),
            max(resolved_limit, cache_len or resolved_limit),
        )

        vector_task = asyncio.create_task(backend._retrieve_vector_candidates(query, candidate_k))
        entity_task = asyncio.create_task(backend._retrieve_entity_candidates(query))
        keyword_task = (
            asyncio.create_task(backend.keyword_extractor.extract(query))
            if self.enable_rrf_signal_rerank and backend.keyword_extractor
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
                key=lambda memory: cosine_similarity(memory.embedding, query_embedding),
                reverse=True,
            )

        bm25_candidates: list[tuple[MemoryItem, float]] = []
        if backend.index_manager:
            ranked = backend.index_manager.search(query, k=candidate_k)
            for memory_id, score in ranked:
                cached_memory = backend.index_manager.get_memory(memory_id)
                if cached_memory:
                    bm25_candidates.append((cached_memory, score))

        vector_candidates = self._score_candidates(vector_memories, query_embedding)
        entity_candidates = self._score_candidates(entity_memories, query_embedding)

        results = await self.retrieve(
            query=query,
            vector_candidates=vector_candidates,
            entity_candidates=entity_candidates,
            graph_candidates=None,
            bm25_candidates=bm25_candidates,
            top_k=candidate_k,
            apply_reranking=False,
        )

        if scoring:
            results = self._apply_scoring_filter(
                backend=backend, results=results, query_embedding=query_embedding
            )

        if (
            self.enable_rrf_signal_rerank
            and query_embedding
            and self.fusion_strategy == FusionStrategy.RRF
        ):
            query_keywords = await keyword_task if keyword_task else []
            rrf_scores = {result.memory.memory_id: result.final_score for result in results}
            ranked_memories = backend.signal_ranker.rank(
                [result.memory for result in results],
                query_embedding,
                extra_signals={
                    "rrf_scores": rrf_scores,
                    "query_keywords": query_keywords,
                    "entity_memory_ids": entity_memory_ids,
                },
            )
            results = self._reorder_results(results, ranked_memories)

        results = await self._maybe_rerank_results(query, results)

        if limit is not None:
            results = results[:limit]
        memories = [result.memory for result in results]
        if update_access and memories:
            await backend._update_access(memories)

        logger.debug(
            "hybrid recall done returned=%d duration_ms=%.2f",
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return memories

    def _require_backend(self) -> BaseRetriever:
        if self._backend is None:
            raise RuntimeError(
                "HybridRetriever recall requires store and embedder. "
                "Use retrieve() with explicit candidates instead."
            )
        return self._backend

    @staticmethod
    def _score_candidates(
        memories: list[MemoryItem],
        query_embedding: list[float] | None,
    ) -> list[tuple[MemoryItem, float]]:
        if not memories:
            return []
        if not query_embedding:
            return [(memory, 0.0) for memory in memories]
        return [
            (memory, max(0.0, cosine_similarity(memory.embedding, query_embedding)))
            for memory in memories
        ]

    @staticmethod
    def _apply_scoring_filter(
        *,
        backend: BaseRetriever,
        results: list[RetrievalResult],
        query_embedding: list[float] | None,
    ) -> list[RetrievalResult]:
        if not results:
            return results
        memories = backend.memory_scorer.rank(
            [result.memory for result in results],
            query_embedding,
        )
        allowed = {memory.memory_id for memory in memories}
        return [result for result in results if result.memory.memory_id in allowed]

    @staticmethod
    def _reorder_results(
        results: list[RetrievalResult],
        ordered_memories: list[MemoryItem],
    ) -> list[RetrievalResult]:
        if not results:
            return results
        result_map = {result.memory.memory_id: result for result in results}
        return [
            result_map[memory.memory_id]
            for memory in ordered_memories
            if memory.memory_id in result_map
        ]

    async def _maybe_rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not self.reranker or self.rerank_candidates <= 0:
            return results
        rerank_limit = min(self.rerank_candidates, len(results))
        if rerank_limit <= 0:
            return results
        rerank_slice = [result.memory for result in results[:rerank_limit]]
        reranked_tuples = await self.reranker.rerank(query, rerank_slice)
        reranked_ids = [memory.memory_id for memory, _score in reranked_tuples]
        result_map = {result.memory.memory_id: result for result in results}
        reranked = [result_map[memory_id] for memory_id in reranked_ids if memory_id in result_map]
        remaining = [result for result in results if result.memory.memory_id not in reranked_ids]
        return reranked + remaining

    def _fuse_candidates(
        self, stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]]
    ) -> list[RetrievalResult]:
        """
        Fuse candidates from different retrieval stages.

        Args:
            stage_candidates: Dict mapping stages to (memory, score) lists

        Returns:
            List of fused RetrievalResult objects
        """
        if self.fusion_strategy == FusionStrategy.RRF:
            return self._reciprocal_rank_fusion(stage_candidates)
        elif self.fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum_fusion(stage_candidates)
        elif self.fusion_strategy == FusionStrategy.MAX_SCORE:
            return self._max_score_fusion(stage_candidates)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _reciprocal_rank_fusion(
        self, stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]], k: int = 60
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF score = sum over all stages of: 1 / (k + rank_in_stage)
        where k is typically 60, and rank starts at 1.
        """
        memory_candidates = {
            stage: [memory for memory, _score in candidates]
            for stage, candidates in stage_candidates.items()
        }
        fused, scores, stage_scores = rrf_fuse(
            stage_candidates=memory_candidates,
            weights=self.weights,
            rrf_k=k,
        )

        results: list[RetrievalResult] = []
        for memory in fused:
            memory_id = memory.memory_id
            result = RetrievalResult(
                memory=memory,
                scores={},
                final_score=scores.get(memory_id, 0.0),
                retrieval_stages=[],
            )
            for stage, per_stage in stage_scores.items():
                if memory_id in per_stage:
                    result.scores[stage.value] = per_stage[memory_id]
                    result.retrieval_stages.append(stage)
            results.append(result)

        return results

    def _ensure_result(
        self,
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

    def _apply_weighted_score(
        self,
        result: RetrievalResult,
        *,
        stage: RetrievalStage,
        weighted_score: float,
    ) -> None:
        result.scores[stage.value] = weighted_score
        result.final_score += weighted_score
        if stage not in result.retrieval_stages:
            result.retrieval_stages.append(stage)

    def _weighted_sum_fusion(
        self, stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]]
    ) -> list[RetrievalResult]:
        """
        Weighted sum fusion.

        Combines normalized scores from each stage using configured weights.
        """
        memory_map: dict[str, RetrievalResult] = {}

        for stage, candidates in stage_candidates.items():
            weight = self.weights.get(stage, 0.0)

            # Normalize scores to [0, 1] range
            if not candidates:
                continue
            scores = [score for _, score in candidates]
            max_score = max(scores) if scores else 1.0

            for memory, score in candidates:
                normalized_score = score / max_score if max_score > 0 else 0.0
                weighted_score = weight * normalized_score
                result = self._ensure_result(memory_map, memory)
                self._apply_weighted_score(
                    result,
                    stage=stage,
                    weighted_score=weighted_score,
                )

        results = list(memory_map.values())
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _max_score_fusion(
        self, stage_candidates: dict[RetrievalStage, list[tuple[MemoryItem, float]]]
    ) -> list[RetrievalResult]:
        """
        Max score fusion.

        Takes the maximum score across all stages for each memory.
        """
        memory_map: dict[str, RetrievalResult] = {}

        for stage, candidates in stage_candidates.items():
            for memory, score in candidates:
                if memory.memory_id not in memory_map:
                    memory_map[memory.memory_id] = RetrievalResult(
                        memory=memory,
                        scores={},
                        final_score=0.0,
                        retrieval_stages=[],
                    )

                result = memory_map[memory.memory_id]
                result.scores[stage.value] = score
                result.final_score = max(result.final_score, score)
                if stage not in result.retrieval_stages:
                    result.retrieval_stages.append(stage)

        results = list(memory_map.values())
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
