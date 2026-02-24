from __future__ import annotations

from langchain_core.embeddings.embeddings import Embeddings

from ..analysis.keywords import KeywordExtractor
from ..ner.base import BaseNER
from ..rerankers import BaseReranker
from ..store.protocols import MemoryCRUDStore
from ._internal.indexing import IndexManager
from .hybrid_retrieval import FusionStrategy, HybridRetriever
from .retrieval import HybridFusionRetriever, Retriever, VectorEntityRetriever
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
        embedder: Embeddings,
        index_manager: IndexManager | None = None,
    ) -> Retriever:
        return VectorEntityRetriever(
            store=store,
            scoring_system=scoring_system,
            ner=ner,
            keyword_extractor=keyword_extractor,
            embedder=embedder,
            index_manager=index_manager,
        )

    @staticmethod
    def create_hybrid(
        *,
        store: MemoryCRUDStore,
        scoring_system: MemoryScoring,
        ner: BaseNER | None,
        keyword_extractor: KeywordExtractor | None,
        embedder: Embeddings,
        index_manager: IndexManager | None = None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        reranker: BaseReranker | None = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.3,
        entity_weight: float = 0.1,
        graph_weight: float = 0.0,
        rrf_k: int = 60,
        enable_rrf_signal_rerank: bool = False,
        rerank_candidates: int = 50,
        use_fusion_retriever: bool = True,
    ) -> Retriever:
        common_args = {
            "store": store,
            "scoring_system": scoring_system,
            "ner": ner,
            "keyword_extractor": keyword_extractor,
            "embedder": embedder,
            "index_manager": index_manager,
        }
        if use_fusion_retriever:
            return HybridFusionRetriever(
                **common_args,
                rrf_k=rrf_k,
                enable_rrf_signal_rerank=enable_rrf_signal_rerank,
                reranker=reranker,
                rerank_candidates=rerank_candidates,
            )
        return HybridRetriever(
            vector_weight=vector_weight,
            entity_weight=entity_weight,
            graph_weight=graph_weight,
            bm25_weight=bm25_weight,
            fusion_strategy=fusion_strategy,
            reranker=reranker,
            memory_scoring=scoring_system,
            store=store,
            embedder=embedder,
            ner=ner,
            keyword_extractor=keyword_extractor,
            index_manager=index_manager,
            rrf_k=rrf_k,
            enable_rrf_signal_rerank=enable_rrf_signal_rerank,
            rerank_candidates=rerank_candidates,
        )
