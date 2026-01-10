"""
Multi-stage hybrid retrieval pipeline with fusion scoring and cross-encoder reranking.

This module implements a sophisticated retrieval strategy:
1. Vector similarity search (semantic retrieval)
2. Entity/graph-based retrieval (structured knowledge)
3. Fusion scoring (combining multiple signals)
4. Optional cross-encoder reranking (semantic precision)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import MemoryItem
from .retrieval import rrf_fuse
from ..rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker
from .scoring import MemoryScoring

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
            RetrievalStage.GRAPH: graph_weight,
        }
        self._normalize_weights()
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker or NoOpReranker()
        self.memory_scoring = memory_scoring

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
            if candidates:
                scores = [score for _, score in candidates]
                max_score = max(scores) if scores else 1.0

                for memory, score in candidates:
                    normalized_score = score / max_score if max_score > 0 else 0.0
                    weighted_score = weight * normalized_score

                    if memory.memory_id not in memory_map:
                        memory_map[memory.memory_id] = RetrievalResult(
                            memory=memory,
                            scores={},
                            final_score=0.0,
                            retrieval_stages=[],
                        )

                    result = memory_map[memory.memory_id]
                    result.scores[stage.value] = weighted_score
                    result.final_score += weighted_score
                    if stage not in result.retrieval_stages:
                        result.retrieval_stages.append(stage)

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
