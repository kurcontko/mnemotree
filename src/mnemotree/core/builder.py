from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Literal

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel

from ..analysis.keywords import KeywordExtractor
from ..ner.base import BaseNER
from ..store.protocols import MemoryCRUDStore
from .memory import (
    IngestionConfig,
    MemoryCore,
    MemoryMode,
    ModeDefaultsConfig,
    NerConfig,
    RetrievalConfig,
    RetrievalMode,
    ScoringConfig,
)
from .models import MemoryItem
from .scoring import MemoryScoring


class MemoryCoreBuilder:
    """Fluent builder for configuring MemoryCore without a long constructor signature."""

    def __init__(self, store: MemoryCRUDStore) -> None:
        self._store = store
        self._llm: BaseLanguageModel | None = None
        self._embeddings: Embeddings | None = None
        self._mode_defaults = ModeDefaultsConfig()
        self._ner_config = NerConfig()
        self._scoring_config = ScoringConfig()
        self._retrieval_config = RetrievalConfig()
        self._ingestion_config = IngestionConfig()

    @classmethod
    def lite(cls, store: MemoryCRUDStore) -> MemoryCoreBuilder:
        return cls(store).with_mode("lite")

    @classmethod
    def pro(cls, store: MemoryCRUDStore) -> MemoryCoreBuilder:
        return cls(store).with_mode("pro")

    def with_mode(self, mode: MemoryMode) -> MemoryCoreBuilder:
        self._mode_defaults = replace(self._mode_defaults, mode=mode)
        return self

    def with_llm(self, llm: BaseLanguageModel | None) -> MemoryCoreBuilder:
        self._llm = llm
        return self

    def with_embeddings(self, embeddings: Embeddings | None) -> MemoryCoreBuilder:
        self._embeddings = embeddings
        return self

    def with_default_importance(self, default_importance: float) -> MemoryCoreBuilder:
        self._scoring_config = replace(
            self._scoring_config,
            default_importance=default_importance,
        )
        return self

    def with_pre_remember_hooks(
        self,
        hooks: list[Callable[[MemoryItem], Awaitable[MemoryItem]]] | None,
    ) -> MemoryCoreBuilder:
        self._scoring_config = replace(self._scoring_config, pre_remember_hooks=hooks)
        return self

    def with_memory_scoring(self, memory_scoring: MemoryScoring | None) -> MemoryCoreBuilder:
        self._scoring_config = replace(self._scoring_config, memory_scoring=memory_scoring)
        return self

    def with_ner(self, ner: BaseNER | None, *, enable: bool = True) -> MemoryCoreBuilder:
        self._ner_config = replace(self._ner_config, ner=ner, enable_ner=enable)
        return self

    def disable_ner(self) -> MemoryCoreBuilder:
        return self.with_ner(None, enable=False)

    def with_keyword_extractor(
        self,
        extractor: KeywordExtractor | None,
        *,
        enable: bool | None = None,
    ) -> MemoryCoreBuilder:
        enable_keywords = self._mode_defaults.enable_keywords
        if enable is not None:
            enable_keywords = enable
        elif extractor is not None:
            enable_keywords = True
        self._mode_defaults = replace(
            self._mode_defaults,
            keyword_extractor=extractor,
            enable_keywords=enable_keywords,
        )
        return self

    def enable_keywords(self, *, extractor: KeywordExtractor | None = None) -> MemoryCoreBuilder:
        self._mode_defaults = replace(self._mode_defaults, enable_keywords=True)
        if extractor is not None:
            self._mode_defaults = replace(self._mode_defaults, keyword_extractor=extractor)
        return self

    def disable_keywords(self) -> MemoryCoreBuilder:
        return self.with_keyword_extractor(None, enable=False)

    def use_retrieval_mode(self, retrieval_mode: RetrievalMode) -> MemoryCoreBuilder:
        self._retrieval_config = replace(
            self._retrieval_config,
            retrieval_mode=retrieval_mode,
        )
        return self

    def use_basic_retrieval(self) -> MemoryCoreBuilder:
        self._retrieval_config = replace(self._retrieval_config, retrieval_mode="basic")
        return self

    def use_hybrid_fusion(
        self,
        *,
        rrf_k: int = 60,
        enable_rrf_signal_rerank: bool = False,
        reranker_backend: Literal["none", "flashrank"] = "none",
        reranker_model: str = "ms-marco-TinyBERT-L-2-v2",
        rerank_candidates: int = 50,
    ) -> MemoryCoreBuilder:
        self._retrieval_config = replace(
            self._retrieval_config,
            retrieval_mode="hybrid",
            rrf_k=rrf_k,
            enable_rrf_signal_rerank=enable_rrf_signal_rerank,
            reranker_backend=reranker_backend,
            reranker_model=reranker_model,
            rerank_candidates=rerank_candidates,
        )
        return self

    def enable_bm25(self, *, k1: float = 1.2, b: float = 0.75) -> MemoryCoreBuilder:
        self._retrieval_config = replace(
            self._retrieval_config,
            enable_bm25=True,
            bm25_k1=k1,
            bm25_b=b,
        )
        return self

    def disable_bm25(self) -> MemoryCoreBuilder:
        self._retrieval_config = replace(self._retrieval_config, enable_bm25=False)
        return self

    def enable_prf(self, *, docs: int = 5, terms: int = 8) -> MemoryCoreBuilder:
        self._retrieval_config = replace(
            self._retrieval_config,
            enable_prf=True,
            prf_docs=docs,
            prf_terms=terms,
        )
        return self

    def disable_prf(self) -> MemoryCoreBuilder:
        self._retrieval_config = replace(self._retrieval_config, enable_prf=False)
        return self

    def with_defaults(
        self,
        *,
        default_analyze: bool | None = None,
        default_summarize: bool | None = None,
    ) -> MemoryCoreBuilder:
        self._mode_defaults = replace(
            self._mode_defaults,
            default_analyze=default_analyze,
            default_summarize=default_summarize,
        )
        return self

    def with_option(self, name: str, value: Any) -> MemoryCoreBuilder:
        if name == "mode_defaults" and isinstance(value, ModeDefaultsConfig):
            self._mode_defaults = value
            return self
        if name == "ner_config" and isinstance(value, NerConfig):
            self._ner_config = value
            return self
        if name == "scoring_config" and isinstance(value, ScoringConfig):
            self._scoring_config = value
            return self
        if name == "retrieval_config" and isinstance(value, RetrievalConfig):
            self._retrieval_config = value
            return self
        if name == "ingestion_config" and isinstance(value, IngestionConfig):
            self._ingestion_config = value
            return self
        if name == "llm":
            self._llm = value
            return self
        if name == "embeddings":
            self._embeddings = value
            return self
        if name == "mode":
            self._mode_defaults = replace(self._mode_defaults, mode=value)
            return self
        if name == "default_analyze":
            self._mode_defaults = replace(self._mode_defaults, default_analyze=value)
            return self
        if name == "default_summarize":
            self._mode_defaults = replace(self._mode_defaults, default_summarize=value)
            return self
        if name == "enable_keywords":
            self._mode_defaults = replace(self._mode_defaults, enable_keywords=value)
            return self
        if name == "keyword_extractor":
            self._mode_defaults = replace(self._mode_defaults, keyword_extractor=value)
            return self
        if name == "ner":
            self._ner_config = replace(self._ner_config, ner=value)
            return self
        if name == "enable_ner":
            self._ner_config = replace(self._ner_config, enable_ner=value)
            return self
        if name == "default_importance":
            self._scoring_config = replace(self._scoring_config, default_importance=value)
            return self
        if name == "pre_remember_hooks":
            self._scoring_config = replace(self._scoring_config, pre_remember_hooks=value)
            return self
        if name == "memory_scoring":
            self._scoring_config = replace(self._scoring_config, memory_scoring=value)
            return self
        if name == "retrieval_mode":
            self._retrieval_config = replace(self._retrieval_config, retrieval_mode=value)
            return self
        if name == "enable_bm25":
            self._retrieval_config = replace(self._retrieval_config, enable_bm25=value)
            return self
        if name == "bm25_k1":
            self._retrieval_config = replace(self._retrieval_config, bm25_k1=value)
            return self
        if name == "bm25_b":
            self._retrieval_config = replace(self._retrieval_config, bm25_b=value)
            return self
        if name == "rrf_k":
            self._retrieval_config = replace(self._retrieval_config, rrf_k=value)
            return self
        if name == "enable_prf":
            self._retrieval_config = replace(self._retrieval_config, enable_prf=value)
            return self
        if name == "prf_docs":
            self._retrieval_config = replace(self._retrieval_config, prf_docs=value)
            return self
        if name == "prf_terms":
            self._retrieval_config = replace(self._retrieval_config, prf_terms=value)
            return self
        if name == "enable_rrf_signal_rerank":
            self._retrieval_config = replace(
                self._retrieval_config,
                enable_rrf_signal_rerank=value,
            )
            return self
        if name == "reranker_backend":
            self._retrieval_config = replace(self._retrieval_config, reranker_backend=value)
            return self
        if name == "reranker_model":
            self._retrieval_config = replace(self._retrieval_config, reranker_model=value)
            return self
        if name == "rerank_candidates":
            self._retrieval_config = replace(self._retrieval_config, rerank_candidates=value)
            return self
        if name == "async_ingest":
            self._ingestion_config = replace(self._ingestion_config, async_ingest=value)
            return self
        if name == "ingestion_queue_size":
            self._ingestion_config = replace(self._ingestion_config, ingestion_queue_size=value)
            return self
        raise ValueError(f"Unknown MemoryCore option: {name}")

    def build(self) -> MemoryCore:
        return MemoryCore(
            store=self._store,
            llm=self._llm,
            embeddings=self._embeddings,
            mode_defaults=self._mode_defaults,
            ner_config=self._ner_config,
            scoring_config=self._scoring_config,
            retrieval_config=self._retrieval_config,
            ingestion_config=self._ingestion_config,
        )
