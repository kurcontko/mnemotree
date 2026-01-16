from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from langchain_core.embeddings.embeddings import Embeddings

from ..analysis.keywords import KeywordExtractor
from ..ner.base import BaseNER
from ..rerankers import BaseReranker
from ..store.protocols import (
    MemoryCRUDStore,
    SupportsEntityQuery,
    SupportsMetadataUpdate,
    SupportsStructuredQuery,
    SupportsVectorSearch,
)
from ._internal.indexing import IndexManager
from ._internal.scoring import MemoryScorer, SignalRanker, cosine_similarity
from .models import MemoryItem
from .query import MemoryQuery, MemoryQueryBuilder
from .scoring import MemoryScoring

logger = logging.getLogger(__name__)

# Backwards-compatible re-exports for the hybrid retrieval API.
_HYBRID_EXPORTS = {
    "BaseReranker",
    "CrossEncoderReranker",
    "FusionStrategy",
    "HybridRetriever",
    "NoOpReranker",
    "RetrievalResult",
    "RetrievalStage",
}


def __getattr__(name: str) -> Any:
    if name in _HYBRID_EXPORTS:
        from . import hybrid_retrieval as _hybrid

        return getattr(_hybrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _HYBRID_EXPORTS)

__all__ = [
    "Retriever",
    "BaseRetriever",
    "VectorEntityRetriever",
    "HybridFusionRetriever",
    "rrf_fuse",
]


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
        embedder: Embeddings,
        index_manager: IndexManager | None = None,
    ) -> None:
        self.store = store
        self.scoring_system = scoring_system
        self.memory_scorer = MemoryScorer(scoring_system)
        self.signal_ranker = SignalRanker()
        self.ner = ner
        self.keyword_extractor = keyword_extractor
        self.embedder = embedder
        self.index_manager = index_manager

    async def _get_embedding(self, text: str) -> list[float]:
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
            memories, query_embedding, query_keywords, entity_memory_ids = (
                await self._recall_from_text(query, limit)
            )
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

        filtered = [memory for similarity, memory in scored_entities if similarity >= min_entity_similarity]
        entity_memory_ids = {memory.memory_id for memory in filtered}
        return filtered, entity_memory_ids


class HybridFusionRetriever(BaseRetriever):
    def __init__(
        self,
        *args: Any,
        rrf_k: int = 60,
        enable_rrf_signal_rerank: bool = False,
        reranker: BaseReranker | None = None,
        rerank_candidates: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rrf_k = rrf_k
        self.enable_rrf_signal_rerank = enable_rrf_signal_rerank
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]:
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
            "rrf recall start limit=%s resolved_limit=%d scoring=%s rerank=%s",
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
        ) = await self._collect_rrf_candidates(query, candidate_k)

        memories, rrf_scores = self._rrf_fuse_with_scores(
            vector_memories=vector_memories,
            entity_memories=entity_memories,
            bm25_memories=bm25_memories,
            rrf_k=self.rrf_k,
        )

        if scoring:
            memories = self.memory_scorer.rank(
                memories,
                query_embedding,
            )

        if self.enable_rrf_signal_rerank and query_embedding:
            query_keywords = await keyword_task if keyword_task else []
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
            "rrf recall done returned=%d duration_ms=%.2f",
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )

        return memories

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
            memories = self.memory_scorer.rank(
                memories,
                query_embedding,
            )

        if limit is not None:
            memories = memories[:limit]
        if update_access and memories:
            await self._update_access(memories)

        logger.debug(
            "rrf recall done (structured) returned=%d duration_ms=%.2f",
            len(memories),
            (time.perf_counter() - started) * 1000.0,
        )
        return memories

    async def _collect_rrf_candidates(
        self, query: str, candidate_k: int
    ) -> tuple[
        list[MemoryItem],
        list[MemoryItem],
        list[MemoryItem],
        list[float] | None,
        set[str],
        asyncio.Task[list[str]] | None,
    ]:
        vector_task = asyncio.create_task(
            self._retrieve_vector_candidates(query, candidate_k)
        )
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
                key=lambda memory: cosine_similarity(memory.embedding, query_embedding),
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
            "rrf recall staged vector=%d entity=%d bm25=%d candidate_k=%d",
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

    async def _maybe_rerank(self, query: str, memories: list[MemoryItem]) -> list[MemoryItem]:
        if not self.reranker or self.rerank_candidates <= 0:
            return memories
        rerank_limit = min(self.rerank_candidates, len(memories))
        if rerank_limit <= 0:
            return memories
        rerank_slice = memories[:rerank_limit]
        reranked_tuples = await self.reranker.rerank(query, rerank_slice)
        reranked = [memory for memory, _score in reranked_tuples]
        return reranked + memories[rerank_limit:]

    def _rrf_fuse_with_scores(
        self,
        *,
        vector_memories: list[MemoryItem],
        entity_memories: list[MemoryItem],
        bm25_memories: list[MemoryItem],
        rrf_k: int,
    ) -> tuple[list[MemoryItem], dict[str, float]]:
        weights = {"vector": 0.6, "bm25": 0.3, "entity": 0.1}
        fused, scores, _stage_scores = rrf_fuse(
            stage_candidates={
                "vector": vector_memories,
                "bm25": bm25_memories,
                "entity": entity_memories,
            },
            weights=weights,
            rrf_k=rrf_k,
        )
        return fused, scores
