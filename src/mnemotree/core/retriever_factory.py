from __future__ import annotations

from typing import Any

from ..analysis.keywords import KeywordExtractor
from ..ner.base import BaseNER
from ..rerankers import BaseReranker
from ..store.protocols import MemoryCRUDStore
from ._internal.indexing import IndexManager
from .protocols import EmbeddingModel
from .retrieval import FusionStrategy, HybridRetriever, Retriever, VectorEntityRetriever
from .scoring import MemoryScoring


class RetrieverFactory:
    """Factory for constructing retrievers with shared dependencies."""

    @staticmethod
    def create_basic(
        *,
        store: MemoryCRUDStore,
        scoring_system: MemoryScoring,
        ner: BaseNER | None,
        keyword_extractor: KeywordExtractor | None,
        embedder: EmbeddingModel,
        index_manager: IndexManager | None = None,
        hyde_embedder: Any = None,
    ) -> Retriever:
        return VectorEntityRetriever(
            store=store,
            scoring_system=scoring_system,
            ner=ner,
            keyword_extractor=keyword_extractor,
            embedder=embedder,
            index_manager=index_manager,
            hyde_embedder=hyde_embedder,
        )

    @staticmethod
    def create_hybrid(
        *,
        store: MemoryCRUDStore,
        scoring_system: MemoryScoring,
        ner: BaseNER | None,
        keyword_extractor: KeywordExtractor | None,
        embedder: EmbeddingModel,
        index_manager: IndexManager | None = None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        reranker: BaseReranker | None = None,
        vector_weight: float = 0.50,
        bm25_weight: float = 0.35,
        entity_weight: float = 0.15,
        graph_weight: float = 0.0,
        rrf_k: int = 60,
        enable_rrf_signal_rerank: bool = False,
        rerank_candidates: int = 50,
        hyde_embedder: Any = None,
        # Backward compat — ignored
        use_fusion_retriever: bool = True,
    ) -> Retriever:
        return HybridRetriever(
            store=store,
            scoring_system=scoring_system,
            ner=ner,
            keyword_extractor=keyword_extractor,
            embedder=embedder,
            index_manager=index_manager,
            hyde_embedder=hyde_embedder,
            vector_weight=vector_weight,
            entity_weight=entity_weight,
            bm25_weight=bm25_weight,
            graph_weight=graph_weight,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
            enable_rrf_signal_rerank=enable_rrf_signal_rerank,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )
