import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..analysis.clustering import ClusteringResult, MemoryClusterer
from ..analysis.keywords import KeywordExtractor, SpacyKeywordExtractor
from ..analysis.memory_analyzer import MemoryAnalyzer
from ..analysis.models import MemoryAnalysisResult
from ..analysis.summarizer import Summarizer
from ..embeddings.local import LocalSentenceTransformerEmbeddings
from ..ner.base import BaseNER
from ..ner.spacy import SpacyNER
from ..rerankers import FlashRankReranker
from ..store.base import BaseMemoryStore
from ..store.protocols import (
    MemoryCRUDStore,
    SupportsConnections,
    SupportsStructuredQuery,
    SupportsVectorSearch,
)
from ..store.serialization import serialize_datetime
from ._internal.enrichment import EnrichmentResult, StandardEnrichmentPipeline
from ._internal.indexing import IndexManager
from ._internal.ingestion_queue import IngestionRequest, MemoryIngestionQueue
from ._internal.persistence import DefaultPersistence
from .models import MemoryItem, MemoryType
from .query import FilterOperator, MemoryFilter, MemoryQuery, MemoryQueryBuilder
from .retrieval import HybridFusionRetriever, Retriever, VectorEntityRetriever
from .scoring import MemoryScoring

MemoryMode = Literal["lite", "pro"]
RetrievalMode = Literal["basic", "hybrid"]


@dataclass(frozen=True)
class ModeDefaultsConfig:
    mode: MemoryMode = "lite"
    default_analyze: bool | None = None
    default_summarize: bool | None = None
    enable_keywords: bool | None = None
    keyword_extractor: KeywordExtractor | None = None


@dataclass(frozen=True)
class NerConfig:
    ner: BaseNER | None = None
    enable_ner: bool = True


@dataclass(frozen=True)
class ScoringConfig:
    default_importance: float = 0.5
    pre_remember_hooks: list[Callable[[MemoryItem], Awaitable[MemoryItem]]] | None = None
    memory_scoring: MemoryScoring | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_mode: RetrievalMode = "basic"
    enable_bm25: bool = False
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    enable_prf: bool = False
    prf_docs: int = 5
    prf_terms: int = 8
    enable_rrf_signal_rerank: bool = False
    reranker_backend: Literal["none", "flashrank"] = "none"
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    rerank_candidates: int = 50


@dataclass(frozen=True)
class IngestionConfig:
    async_ingest: bool = False
    ingestion_queue_size: int = 100


@dataclass(frozen=True)
class ModeConfig:
    analyze_default: bool
    summarize_default: bool
    enable_keywords: bool


class MemoryCore:
    """Core memory management system."""

    def __init__(
        self,
        store: MemoryCRUDStore,
        llm: BaseLanguageModel | None = None,
        embeddings: Embeddings | None = None,
        *,
        mode_defaults: ModeDefaultsConfig | None = None,
        ner_config: NerConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
        ingestion_config: IngestionConfig | None = None,
    ):
        """
        Initializes the MemoryCore.

        Args:
            store: The underlying storage for memory items.
            llm: Optional Language model for analysis if enabled.
            embeddings: Optional embeddings model. Defaults depend on mode if not provided.
            mode_defaults: Mode defaults and keyword extraction settings.
            ner_config: Named entity recognition configuration.
            scoring_config: Scoring defaults and pre-remember hooks.
            retrieval_config: Retrieval and reranking configuration.
            ingestion_config: Async ingestion configuration.
        """
        self.store = store

        mode_defaults = mode_defaults or ModeDefaultsConfig()
        ner_config = ner_config or NerConfig()
        scoring_config = scoring_config or ScoringConfig()
        retrieval_config = retrieval_config or RetrievalConfig()
        ingestion_config = ingestion_config or IngestionConfig()

        self._init_runtime_settings(
            mode=mode_defaults.mode,
            retrieval_mode=retrieval_config.retrieval_mode,
            enable_bm25=retrieval_config.enable_bm25,
            rrf_k=retrieval_config.rrf_k,
            enable_prf=retrieval_config.enable_prf,
            prf_docs=retrieval_config.prf_docs,
            prf_terms=retrieval_config.prf_terms,
            enable_rrf_signal_rerank=retrieval_config.enable_rrf_signal_rerank,
            reranker_backend=retrieval_config.reranker_backend,
            reranker_model=retrieval_config.reranker_model,
            rerank_candidates=retrieval_config.rerank_candidates,
            async_ingest=ingestion_config.async_ingest,
            ingestion_queue_size=ingestion_config.ingestion_queue_size,
        )
        self._init_mode_defaults(
            mode=mode_defaults.mode,
            default_analyze=mode_defaults.default_analyze,
            default_summarize=mode_defaults.default_summarize,
            enable_keywords=mode_defaults.enable_keywords,
            keyword_extractor=mode_defaults.keyword_extractor,
        )
        self._init_embeddings_and_analysis(
            llm=llm,
            embeddings=embeddings,
        )
        self._init_ner(ner=ner_config.ner, enable_ner=ner_config.enable_ner)
        self._init_scoring_and_hooks(
            memory_scoring=scoring_config.memory_scoring,
            default_importance=scoring_config.default_importance,
            pre_remember_hooks=scoring_config.pre_remember_hooks,
        )
        self._init_components(
            enable_bm25=retrieval_config.enable_bm25,
            enable_prf=retrieval_config.enable_prf,
            bm25_k1=retrieval_config.bm25_k1,
            bm25_b=retrieval_config.bm25_b,
            rrf_k=retrieval_config.rrf_k,
            prf_docs=retrieval_config.prf_docs,
            prf_terms=retrieval_config.prf_terms,
            enable_rrf_signal_rerank=retrieval_config.enable_rrf_signal_rerank,
            rerank_candidates=retrieval_config.rerank_candidates,
        )

    async def remember(
        self,
        content: str,
        *,
        memory_type: MemoryType | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
        analyze: bool | None = None,
        summarize: bool | None = None,
        references: list[str] | None = None,
        skip_store: bool = False,
    ) -> MemoryItem:
        """Store a new memory with optional analysis - delegating to EnrichmentPipeline."""
        if self.async_ingest and not skip_store:
            return await self.remember_async(
                content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                context=context,
                analyze=analyze,
                summarize=summarize,
                references=references,
            )
        return await self._remember_sync(
            content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            context=context,
            analyze=analyze,
            summarize=summarize,
            references=references,
            skip_store=skip_store,
        )

    async def remember_async(
        self,
        content: str,
        *,
        memory_type: MemoryType | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
        analyze: bool | None = None,
        summarize: bool | None = None,
        references: list[str] | None = None,
    ) -> MemoryItem:
        """Queue memory ingestion for background processing."""
        await self._ensure_ingestion_queue()
        queue = self._ingestion_queue
        assert queue is not None
        memory_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)
        await queue.enqueue(
            IngestionRequest(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                context=context,
                analyze=analyze,
                summarize=summarize,
                references=references,
                timestamp=timestamp,
            )
        )
        return self._queued_memory_stub(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            context=context,
            timestamp=timestamp,
        )

    async def _remember_sync(
        self,
        content: str,
        *,
        memory_type: MemoryType | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
        analyze: bool | None = None,
        summarize: bool | None = None,
        references: list[str] | None = None,
        skip_store: bool = False,
        memory_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryItem:
        analyze, summarize = self._resolve_analysis_flags(analyze, summarize)

        enrichment = await self.enrichment.enrich(
            content, context, analyze=analyze, summarize=summarize
        )

        memory_type, importance = self._resolve_importance_and_type(
            memory_type,
            importance,
            enrichment.analysis,
        )
        all_tags = self._resolve_tags(tags, enrichment.analysis, enrichment.keywords)

        emotions, emotional_valence, emotional_arousal = self._extract_emotions(enrichment)
        memory_data = self._build_memory_data(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=all_tags,
            context=context,
            enrichment=enrichment,
            emotions=emotions,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            timestamp=timestamp,
        )

        memory = MemoryItem(**memory_data)
        memory = await self._apply_pre_remember_hooks(memory)
        await self._persist_memory(memory, references, skip_store)
        return memory

    def _resolve_analysis_flags(
        self,
        analyze: bool | None,
        summarize: bool | None,
    ) -> tuple[bool, bool]:
        if analyze is None:
            analyze = self.default_analyze
        if summarize is None:
            summarize = self.default_summarize
        return analyze, summarize

    @staticmethod
    def _extract_emotions(
        enrichment: EnrichmentResult,
    ) -> tuple[list[str], float | None, float | None]:
        emotions: list[str] = []
        emotional_valence = None
        emotional_arousal = None
        if enrichment.analysis:
            if enrichment.analysis.emotions:
                emotions = [str(e) for e in enrichment.analysis.emotions]
            emotional_valence = enrichment.analysis.emotional_valence
            emotional_arousal = enrichment.analysis.emotional_arousal
        return emotions, emotional_valence, emotional_arousal

    def _build_memory_data(
        self,
        *,
        memory_id: str | None,
        content: str,
        memory_type: MemoryType,
        importance: float,
        tags: list[str],
        context: dict[str, Any] | None,
        enrichment: EnrichmentResult,
        emotions: list[str],
        emotional_valence: float | None,
        emotional_arousal: float | None,
        timestamp: datetime | None,
    ) -> dict[str, Any]:
        data = {
            "memory_id": memory_id or str(uuid4()),
            "content": content,
            "summary": enrichment.summary,
            "memory_type": memory_type,
            "importance": importance,
            "tags": list(tags),
            "context": context or {},
            "embedding": enrichment.embedding,
            "emotions": emotions,
            "emotional_valence": emotional_valence,
            "emotional_arousal": emotional_arousal,
            "linked_concepts": (
                list(enrichment.analysis.linked_concepts)
                if enrichment.analysis and enrichment.analysis.linked_concepts
                else []
            ),
            "entities": enrichment.entities or {},
            "entity_mentions": enrichment.entity_mentions or {},
        }
        if timestamp is not None:
            data["timestamp"] = serialize_datetime(timestamp) if isinstance(timestamp, datetime) else timestamp
        return data

    async def _apply_pre_remember_hooks(self, memory: MemoryItem) -> MemoryItem:
        if not self.pre_remember_hooks:
            return memory
        for hook in self.pre_remember_hooks:
            memory = await hook(memory)
        return memory

    async def _persist_memory(
        self,
        memory: MemoryItem,
        references: list[str] | None,
        skip_store: bool,
    ) -> None:
        if skip_store:
            return
        store_tasks = [self.persistence.save(memory)]
        if references:
            store_tasks.append(self.connect(memory.memory_id, related_to=references))
        await asyncio.gather(*store_tasks)

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        *,
        limit: int | None = None,
        scoring: bool = True,
        update_access: bool = False,
    ) -> list[MemoryItem]:
        """
        Retrieve memories based on a query - delegating to the retriever.

        Args:
            query: The query string, MemoryQuery object or MemoryQueryBuilder
            limit: Optional max results to return
            update_access: If True, update access metadata for returned memories

        Returns:
            List of matching MemoryItems
        """
        return await self.retrieval.recall(
            query=query, limit=limit, scoring=scoring, update_access=update_access
        )

    async def reflect(
        self, query_builder: MemoryQueryBuilder | None = None, min_importance: float = 0.7
    ) -> dict[str, Any]:
        """
        Analyze patterns and insights across memories.

        Args:
            query_builder: Optional MemoryQueryBuilder for filtering memories.
            min_importance: Minimum importance threshold

        Returns:
            Analysis results including patterns and insights
        """
        if not self.analyzer:
            raise RuntimeError("Analyzer not configured. Use mode='pro' or provide an analyzer.")

        if query_builder is None:
            query_builder = MemoryQueryBuilder().importance_range(min_value=min_importance)
        else:
            query_builder.importance_range(min_value=min_importance)

        if not isinstance(self.store, SupportsStructuredQuery):
            raise NotImplementedError(
                "This store does not support structured queries (query_memories)."
            )

        memories = await self.store.query_memories(await query_builder.build())

        if not memories:
            return {"patterns": [], "insights": [], "summary": "No memories found for analysis"}

        # Analyze patterns and connections
        memory_texts = [
            f"Memory {i + 1}: {m.content} (Type: {m.memory_type.value}, "
            f"Importance: {m.importance:.2f})"
            for i, m in enumerate(memories)
        ]

        analysis = await self.analyzer.analyze_patterns("\n".join(memory_texts))
        return analysis

    async def connect(
        self,
        memory_id: str,
        *,
        related_to: list[str] | None = None,
        conflicts_with: list[str] | None = None,
        previous: str | None = None,
        next: str | None = None,
    ) -> None:
        """
        Create connections between memories.

        Args:
            memory_id: Source memory ID
            related_to: IDs of related memories
            conflicts_with: IDs of conflicting memories
            previous: ID of previous memory in sequence
            next: ID of next memory in sequence
        """
        if not isinstance(self.store, SupportsConnections):
            raise NotImplementedError(
                "This store does not support graph connections (update_connections)."
            )

        await self.store.update_connections(
            memory_id=memory_id,
            related_ids=related_to,
            conflict_ids=conflicts_with,
            previous_id=previous,
            next_id=next,
        )

    async def forget(self, memory_id: str, *, cascade: bool = False) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: ID of memory to delete
            cascade: Whether to delete connected memories

        Returns:
            Success status
        """
        return await self.persistence.delete(memory_id, cascade=cascade)

    async def summarize(
        self, memories: list[MemoryItem], format: str = "text"
    ) -> str | dict[str, Any]:
        """
        Generate a summary of multiple memories.

        Args:
            memories: List of memories to summarize
            format: Output format ("text" or "structured")

        Returns:
            Summary in requested format
        """
        if not self.summarizer:
            raise RuntimeError("Summarizer not configured. Use mode='pro' or provide a summarizer.")

        if not memories:
            return "" if format == "text" else {}

        memory_texts = [f"- {m.content}" for m in memories]
        return await self.summarizer.summarize("\n".join(memory_texts), format=format)

    async def batch_remember(
        self,
        contents: list[str],
        *,
        analyze: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """
        Store multiple memories efficiently.

        Args:
            contents: List of memory contents
            analyze: Whether to analyze contents (defaults based on mode)
            context: Optional shared context

        Returns:
            List of created memories
        """
        tasks = [self.remember(content, analyze=analyze, context=context) for content in contents]
        return await asyncio.gather(*tasks)

    async def search(
        self, query: str, *, filters: dict[str, Any] | None = None, limit: int = 10
    ) -> list[MemoryItem]:
        """
        Search memories with optional filtering.

        Args:
            query: Search query
            filters: Optional filters to apply
            limit: Max results to return

        Returns:
            List of memory items
        """
        # Prefer lexical search when BM25 is enabled and no filters are requested.
        # This keeps MemoryCore as a thin facade and routes lexical concerns to IndexManager.
        bm25_results = await self._maybe_bm25_search(query, limit=limit, filters=filters)
        if bm25_results is not None:
            return bm25_results

        query_embedding = await self.get_embedding(query)
        filter_list = self._build_filter_list(filters)

        supports_structured = self._supports_structured_query()
        if filter_list and not supports_structured:
            raise NotImplementedError(
                "This store does not support structured queries with filters."
            )

        if supports_structured:
            return await self._query_structured(query_embedding, filter_list, limit)

        if not filter_list and isinstance(self.store, SupportsVectorSearch):
            return await self._query_vector(query, query_embedding, limit)

        raise NotImplementedError("This store does not support search().")

    async def _maybe_bm25_search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[MemoryItem] | None:
        if (
            not self.index_manager
            or not getattr(self.index_manager, "enable_bm25", False)
            or filters
            or self.index_manager.doc_count <= 0
        ):
            return None

        ranked = self.index_manager.search(query, k=limit)
        if not ranked:
            return None

        memories: list[MemoryItem] = []
        for memory_id, _score in ranked:
            memory = self.index_manager.get_memory(memory_id)
            if memory is None:
                memory = await self.store.get_memory(memory_id)
            if memory is not None:
                memories.append(memory)
                if len(memories) >= limit:
                    break

        return memories or None

    @staticmethod
    def _build_filter_list(filters: dict[str, Any] | None) -> list[MemoryFilter]:
        if not filters:
            return []
        return [
            MemoryFilter(field=field, operator=FilterOperator.EQ, value=value)
            for field, value in filters.items()
        ]

    async def _query_structured(
        self,
        query_embedding: Any,
        filter_list: list[MemoryFilter],
        limit: int,
    ) -> list[MemoryItem]:
        query = MemoryQuery(
            filters=filter_list,
            vector=query_embedding,
            limit=limit,
        )
        store = cast(SupportsStructuredQuery, self.store)
        return await store.query_memories(query)

    def _supports_structured_query(self) -> bool:
        if not isinstance(self.store, SupportsStructuredQuery):
            return False
        if isinstance(self.store, BaseMemoryStore):
            return type(self.store).query_memories is not BaseMemoryStore.query_memories
        return True

    async def _query_vector(
        self,
        query: str,
        query_embedding: Any,
        limit: int,
    ) -> list[MemoryItem]:
        if hasattr(self.store, 'get_similar_memories'):
            return await self.store.get_similar_memories(
                query=query,
                query_embedding=query_embedding,
                top_k=limit,
                filters=None,
            )
        # Fallback for stores without vector search
        return []

    async def cluster(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder | None = None,
        method: Literal["vector_dbscan", "vector_kmeans", "graph_community"] = "vector_dbscan",
        **clustering_params,
    ) -> tuple[list[MemoryItem], ClusteringResult]:
        """
        Cluster memories based on similarity.

        Args:
            query: Optional query to filter memories before clustering
            method: Clustering method to use
            **clustering_params: Additional parameters for clustering

        Returns:
            Tuple of (memories, clustering_results)
        """
        if not self.clusterer:
            raise RuntimeError("Clusterer not configured. Use mode='pro' or provide a summarizer.")

        # Get memories to cluster
        if query is not None:
            memories = await self.recall(query)
        else:
            # TODO: Add support for all memories
            raise ValueError("Query is required for clustering")

        # Perform clustering
        results = await self.clusterer.cluster_memories(
            memories, method=method, **clustering_params
        )

        return memories, results

    def decay_and_reinforce(self):
        """
        Decay and reinforce memory importance scores.

        This method should be called periodically to update memory importance scores.
        """
        return None

    async def get_embedding(self, text: str) -> Any:
        """Get the embedding for a given text."""
        if not self.embedder:
            raise RuntimeError("Embedder not configured.")
        return await self.embedder.aembed_query(text)

    def _resolve_mode_config(
        self,
        *,
        mode: MemoryMode,
        default_analyze: bool | None,
        default_summarize: bool | None,
        enable_keywords: bool | None,
        keyword_extractor: KeywordExtractor | None,
    ) -> ModeConfig:
        analyze_default = default_analyze if default_analyze is not None else mode == "pro"
        summarize_default = default_summarize if default_summarize is not None else mode == "pro"
        if enable_keywords is None:
            enable_keywords = mode == "lite"
        if keyword_extractor is not None:
            enable_keywords = True
        return ModeConfig(
            analyze_default=analyze_default,
            summarize_default=summarize_default,
            enable_keywords=enable_keywords,
        )

    def _init_runtime_settings(
        self,
        *,
        mode: MemoryMode,
        retrieval_mode: RetrievalMode,
        enable_bm25: bool,
        rrf_k: int,
        enable_prf: bool,
        prf_docs: int,
        prf_terms: int,
        enable_rrf_signal_rerank: bool,
        reranker_backend: Literal["none", "flashrank"],
        reranker_model: str,
        rerank_candidates: int,
        async_ingest: bool,
        ingestion_queue_size: int,
    ) -> None:
        self.mode = mode
        self.retrieval_mode = retrieval_mode
        self.enable_bm25 = enable_bm25
        self.rrf_k = rrf_k
        self.enable_prf = enable_prf
        self.prf_docs = max(0, int(prf_docs))
        self.prf_terms = max(0, int(prf_terms))
        self.enable_rrf_signal_rerank = enable_rrf_signal_rerank
        self.reranker_backend = reranker_backend
        self.reranker_model = reranker_model
        self.rerank_candidates = max(0, int(rerank_candidates))
        self._flashrank_reranker = (
            FlashRankReranker(model_name=reranker_model)
            if reranker_backend == "flashrank"
            else None
        )
        self.async_ingest = async_ingest
        self.ingestion_queue_size = max(1, int(ingestion_queue_size))
        self._ingestion_queue: MemoryIngestionQueue | None = None

    def _init_mode_defaults(
        self,
        *,
        mode: MemoryMode,
        default_analyze: bool | None,
        default_summarize: bool | None,
        enable_keywords: bool | None,
        keyword_extractor: KeywordExtractor | None,
    ) -> None:
        self.mode_config = self._resolve_mode_config(
            mode=mode,
            default_analyze=default_analyze,
            default_summarize=default_summarize,
            enable_keywords=enable_keywords,
            keyword_extractor=keyword_extractor,
        )
        self.default_analyze = self.mode_config.analyze_default
        self.default_summarize = self.mode_config.summarize_default
        self.enable_keywords = self.mode_config.enable_keywords
        self.keyword_extractor = self._resolve_keyword_extractor(
            keyword_extractor, self.enable_keywords
        )

    def _resolve_keyword_extractor(
        self,
        keyword_extractor: KeywordExtractor | None,
        enable_keywords: bool,
    ) -> KeywordExtractor | None:
        """Resolve keyword extractor based on configuration."""
        if keyword_extractor is not None:
            return keyword_extractor
        if enable_keywords:
            return SpacyKeywordExtractor()
        return None

    def _init_embeddings_and_analysis(
        self,
        *,
        llm: BaseLanguageModel | None,
        embeddings: Embeddings | None,
    ) -> None:
        if embeddings is None:
            embeddings = self._resolve_embeddings(self.mode)
        self.embedder = embeddings
        self.analyzer, self.summarizer = self._resolve_analyzer_and_summarizer(
            llm,
            embeddings,
        )
        self.clusterer = MemoryClusterer(self.summarizer) if self.summarizer else None

    def _init_ner(self, *, ner: BaseNER | None, enable_ner: bool) -> None:
        if enable_ner:
            self.ner: BaseNER | None = ner if ner is not None else SpacyNER()
        else:
            self.ner = None

    def _init_scoring_and_hooks(
        self,
        *,
        memory_scoring: MemoryScoring | None,
        default_importance: float,
        pre_remember_hooks: list[Callable[[MemoryItem], Awaitable[MemoryItem]]] | None,
    ) -> None:
        self.memory_scoring = memory_scoring or MemoryScoring()
        self.default_importance = default_importance
        self.pre_remember_hooks = pre_remember_hooks or []

    def _init_components(
        self,
        *,
        enable_bm25: bool,
        enable_prf: bool,
        bm25_k1: float,
        bm25_b: float,
        rrf_k: int,
        prf_docs: int,
        prf_terms: int,
        enable_rrf_signal_rerank: bool,
        rerank_candidates: int,
    ) -> None:
        # _bm25_index/_bm25_cache are now managed by IndexManager
        # self._bm25_index and self._bm25_cache removed
        self.index_manager = IndexManager(
            enable_bm25=enable_bm25,
            enable_prf=enable_prf,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            rrf_k=rrf_k,
            prf_docs=prf_docs,
            prf_terms=prf_terms,
        )
        self.persistence = DefaultPersistence(self.store, self.index_manager)
        self.enrichment = StandardEnrichmentPipeline(
            embedder=self.embedder,
            ner=self.ner,
            keyword_extractor=self.keyword_extractor,
            analyzer=self.analyzer,
            summarizer=self.summarizer,
        )
        self.retrieval: Retriever = self._build_retriever(
            rrf_k=rrf_k,
            enable_rrf_signal_rerank=enable_rrf_signal_rerank,
            rerank_candidates=rerank_candidates,
        )

    def _build_retriever(
        self,
        *,
        rrf_k: int,
        enable_rrf_signal_rerank: bool,
        rerank_candidates: int,
    ) -> Retriever:
        common_retrieval_args = {
            "store": self.store,
            "scoring_system": self.memory_scoring,
            "ner": self.ner,
            "keyword_extractor": self.keyword_extractor,
            "embedder": self.embedder,
            "index_manager": self.index_manager,
        }
        if self.retrieval_mode == "hybrid":
            return HybridFusionRetriever(
                **common_retrieval_args,
                rrf_k=rrf_k,
                enable_rrf_signal_rerank=enable_rrf_signal_rerank,
                reranker=self._flashrank_reranker,
                rerank_candidates=rerank_candidates,
            )
        return VectorEntityRetriever(**common_retrieval_args)

    def _resolve_embeddings(self, mode: MemoryMode) -> Embeddings:
        if mode == "lite":
            lite_model = os.getenv(
                "MNEMOTREE_LITE_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            return LocalSentenceTransformerEmbeddings(model_name=lite_model)

        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        openai_client_kwargs: dict[str, Any] = (
            {"base_url": openai_base_url} if openai_base_url else {}
        )
        return OpenAIEmbeddings(model=openai_embedding_model, **openai_client_kwargs)

    def _resolve_analyzer_and_summarizer(
        self,
        llm: BaseLanguageModel | None,
        embeddings: Embeddings,
    ) -> tuple[MemoryAnalyzer | None, Summarizer | None]:
        should_enable_llm_defaults = self.default_analyze or self.default_summarize

        if self.mode == "pro" and llm is None and should_enable_llm_defaults:
            openai_base_url = os.getenv("OPENAI_BASE_URL")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            openai_client_kwargs: dict[str, Any] = (
                {"base_url": openai_base_url} if openai_base_url else {}
            )
            llm = ChatOpenAI(model=openai_model, temperature=0, **openai_client_kwargs)
        if llm is None:
            return None, None
        return (
            MemoryAnalyzer(llm=llm, embeddings=embeddings),
            Summarizer(llm=llm),
        )

    def _resolve_importance_and_type(
        self,
        memory_type: MemoryType | None,
        importance: float | None,
        analysis: MemoryAnalysisResult | None,
    ) -> tuple[MemoryType, float]:
        if analysis:
            resolved_type = memory_type if memory_type is not None else analysis.memory_type
            resolved_importance = importance if importance is not None else analysis.importance
            return resolved_type, resolved_importance

        resolved_type = memory_type if memory_type is not None else MemoryType.SEMANTIC
        resolved_importance = importance if importance is not None else self.default_importance
        return resolved_type, resolved_importance

    def _resolve_tags(
        self,
        tags: list[str] | None,
        analysis: MemoryAnalysisResult | None,
        keyword_tags: list[str],
    ) -> list[str]:
        all_tags = set(tags or [])
        if analysis and analysis.tags:
            all_tags |= set(analysis.tags)
        if keyword_tags:
            all_tags |= set(keyword_tags)
        return list(all_tags)

    async def _ensure_ingestion_queue(self) -> None:
        if self._ingestion_queue is None:
            self._ingestion_queue = MemoryIngestionQueue(
                self._process_ingestion_request,
                maxsize=self.ingestion_queue_size,
            )
            self._ingestion_queue.start()

    async def _process_ingestion_request(self, request: IngestionRequest) -> None:
        await self._remember_sync(
            request.content,
            memory_type=request.memory_type,
            importance=request.importance,
            tags=request.tags,
            context=request.context,
            analyze=request.analyze,
            summarize=request.summarize,
            references=request.references,
            memory_id=request.memory_id,
            timestamp=request.timestamp,
            skip_store=False,
        )

    def _queued_memory_stub(
        self,
        *,
        memory_id: str,
        content: str,
        memory_type: MemoryType | None,
        importance: float | None,
        tags: list[str] | None,
        context: dict[str, Any] | None,
        timestamp: datetime | None,
    ) -> MemoryItem:
        resolved_type, resolved_importance = self._resolve_importance_and_type(
            memory_type,
            importance,
            None,
        )
        return MemoryItem(
            memory_id=memory_id,
            content=content,
            summary=None,
            memory_type=resolved_type,
            importance=resolved_importance,
            tags=tags or [],
            context=context or {},
            metadata={"queued": True},
            timestamp=timestamp or datetime.now(timezone.utc),
        )
