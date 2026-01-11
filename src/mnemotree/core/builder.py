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
        typed_options = {
            "mode_defaults": (ModeDefaultsConfig, lambda v: setattr(self, "_mode_defaults", v)),
            "ner_config": (NerConfig, lambda v: setattr(self, "_ner_config", v)),
            "scoring_config": (ScoringConfig, lambda v: setattr(self, "_scoring_config", v)),
            "retrieval_config": (RetrievalConfig, lambda v: setattr(self, "_retrieval_config", v)),
            "ingestion_config": (IngestionConfig, lambda v: setattr(self, "_ingestion_config", v)),
        }
        option = typed_options.get(name)
        if option and isinstance(value, option[0]):
            option[1](value)
            return self

        simple_setters = {
            "llm": lambda v: setattr(self, "_llm", v),
            "embeddings": lambda v: setattr(self, "_embeddings", v),
        }
        setter = simple_setters.get(name)
        if setter:
            setter(value)
            return self

        mode_defaults_fields = {
            "mode": "mode",
            "default_analyze": "default_analyze",
            "default_summarize": "default_summarize",
            "enable_keywords": "enable_keywords",
            "keyword_extractor": "keyword_extractor",
        }
        mode_field = mode_defaults_fields.get(name)
        if mode_field:
            self._mode_defaults = replace(self._mode_defaults, **{mode_field: value})
            return self

        ner_fields = {"ner": "ner", "enable_ner": "enable_ner"}
        ner_field = ner_fields.get(name)
        if ner_field:
            self._ner_config = replace(self._ner_config, **{ner_field: value})
            return self

        scoring_fields = {
            "default_importance": "default_importance",
            "pre_remember_hooks": "pre_remember_hooks",
            "memory_scoring": "memory_scoring",
        }
        scoring_field = scoring_fields.get(name)
        if scoring_field:
            self._scoring_config = replace(self._scoring_config, **{scoring_field: value})
            return self

        retrieval_fields = {
            "retrieval_mode": "retrieval_mode",
            "enable_bm25": "enable_bm25",
            "bm25_k1": "bm25_k1",
            "bm25_b": "bm25_b",
            "rrf_k": "rrf_k",
            "enable_prf": "enable_prf",
            "prf_docs": "prf_docs",
            "prf_terms": "prf_terms",
            "enable_rrf_signal_rerank": "enable_rrf_signal_rerank",
            "reranker_backend": "reranker_backend",
            "reranker_model": "reranker_model",
            "rerank_candidates": "rerank_candidates",
        }
        retrieval_field = retrieval_fields.get(name)
        if retrieval_field:
            self._retrieval_config = replace(self._retrieval_config, **{retrieval_field: value})
            return self

        ingestion_fields = {
            "async_ingest": "async_ingest",
            "ingestion_queue_size": "ingestion_queue_size",
        }
        ingestion_field = ingestion_fields.get(name)
        if ingestion_field:
            self._ingestion_config = replace(self._ingestion_config, **{ingestion_field: value})
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
