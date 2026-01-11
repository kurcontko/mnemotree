"""Tests for MemoryCoreBuilder fluent API."""

from unittest.mock import MagicMock

import pytest

from mnemotree.core.builder import MemoryCoreBuilder
from mnemotree.core.memory import (
    IngestionConfig,
    MemoryCore,
    ModeDefaultsConfig,
    NerConfig,
    RetrievalConfig,
    ScoringConfig,
)
from mnemotree.core.scoring import MemoryScoring
from mnemotree.store.base import BaseMemoryStore


class MockStore(BaseMemoryStore):
    """Minimal mock store for builder tests."""

    async def store_memory(self, memory):
        """Mock implementation - no-op for testing."""
        pass

    async def get_memory(self, mid):
        return None

    async def delete_memory(self, mid, *, cascade=False):
        return True

    async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
        return []

    async def query_memories(self, query):
        return []

    async def update_connections(self, memory_id, **kwargs):
        """Mock implementation - no-op for testing."""
        pass

    async def query_by_entities(self, entities):
        return []

    async def close(self):
        """Mock implementation - no-op for testing."""
        pass


@pytest.fixture
def mock_store():
    return MockStore()


@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    embeddings.aembed_query = MagicMock(return_value=[0.1] * 1536)
    return embeddings


@pytest.fixture
def mock_llm():
    return MagicMock()


# --- Basic Builder Tests ---


def test_builder_instantiation(mock_store):
    """Test that builder can be instantiated with a store."""
    builder = MemoryCoreBuilder(mock_store)
    assert builder._store is mock_store


def test_builder_build_returns_memory_core(mock_store, mock_embeddings):
    """Test that build() returns a valid MemoryCore instance."""
    core = (
        MemoryCoreBuilder(mock_store)
        .with_embeddings(mock_embeddings)
        .disable_keywords()
        .disable_ner()
        .build()
    )
    assert isinstance(core, MemoryCore)


def test_builder_lite_classmethod(mock_store):
    """Test the lite() classmethod sets mode to lite."""
    builder = MemoryCoreBuilder.lite(mock_store)
    assert builder._mode_defaults.mode == "lite"


def test_builder_pro_classmethod(mock_store):
    """Test the pro() classmethod sets mode to pro."""
    builder = MemoryCoreBuilder.pro(mock_store)
    assert builder._mode_defaults.mode == "pro"


# --- Chaining Tests ---


def test_builder_method_chaining(mock_store, mock_embeddings, mock_llm):
    """Test that all builder methods return self for chaining."""
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_mode("pro")
        .with_llm(mock_llm)
        .with_embeddings(mock_embeddings)
        .with_default_importance(0.7)
        .disable_keywords()
        .disable_ner()
        .enable_bm25()
        .enable_prf()
    )
    core = builder.build()
    assert isinstance(core, MemoryCore)


# --- Mode Configuration Tests ---


def test_with_mode_sets_mode(mock_store):
    """Test with_mode() sets the mode correctly."""
    builder = MemoryCoreBuilder(mock_store).with_mode("lite")
    assert builder._mode_defaults.mode == "lite"

    builder = MemoryCoreBuilder(mock_store).with_mode("pro")
    assert builder._mode_defaults.mode == "pro"


# --- LLM and Embeddings Configuration Tests ---


def test_with_llm(mock_store, mock_llm):
    """Test with_llm() sets the LLM."""
    builder = MemoryCoreBuilder(mock_store).with_llm(mock_llm)
    assert builder._llm is mock_llm


def test_with_embeddings(mock_store, mock_embeddings):
    """Test with_embeddings() sets the embeddings."""
    builder = MemoryCoreBuilder(mock_store).with_embeddings(mock_embeddings)
    assert builder._embeddings is mock_embeddings


# --- Scoring Configuration Tests ---


def test_with_default_importance(mock_store):
    """Test with_default_importance() sets the importance value."""
    builder = MemoryCoreBuilder(mock_store).with_default_importance(0.8)
    assert abs(builder._scoring_config.default_importance - 0.8) < 1e-9


def test_with_memory_scoring(mock_store):
    """Test with_memory_scoring() sets the scoring config."""
    scoring = MemoryScoring(importance_weight=0.5, recency_weight=0.3, relevance_weight=0.2)
    builder = MemoryCoreBuilder(mock_store).with_memory_scoring(scoring)
    assert builder._scoring_config.memory_scoring is scoring


def test_with_pre_remember_hooks(mock_store):
    """Test with_pre_remember_hooks() sets the hooks."""

    async def hook(item):
        return item

    builder = MemoryCoreBuilder(mock_store).with_pre_remember_hooks([hook])
    assert builder._scoring_config.pre_remember_hooks == [hook]


# --- NER Configuration Tests ---


def test_with_ner(mock_store):
    """Test with_ner() sets NER config."""
    mock_ner = MagicMock()
    builder = MemoryCoreBuilder(mock_store).with_ner(mock_ner, enable=True)
    assert builder._ner_config.ner is mock_ner
    assert builder._ner_config.enable_ner is True


def test_disable_ner(mock_store):
    """Test disable_ner() disables NER extraction."""
    builder = MemoryCoreBuilder(mock_store).disable_ner()
    assert builder._ner_config.enable_ner is False


# --- Keyword Extractor Configuration Tests ---


def test_with_keyword_extractor(mock_store):
    """Test with_keyword_extractor() sets the extractor."""
    mock_extractor = MagicMock()
    builder = MemoryCoreBuilder(mock_store).with_keyword_extractor(mock_extractor)
    assert builder._mode_defaults.keyword_extractor is mock_extractor
    assert builder._mode_defaults.enable_keywords is True


def test_with_keyword_extractor_explicit_disable(mock_store):
    """Test with_keyword_extractor() with explicit enable=False."""
    mock_extractor = MagicMock()
    builder = MemoryCoreBuilder(mock_store).with_keyword_extractor(mock_extractor, enable=False)
    assert builder._mode_defaults.keyword_extractor is mock_extractor
    assert builder._mode_defaults.enable_keywords is False


def test_enable_keywords(mock_store):
    """Test enable_keywords() enables keyword extraction."""
    builder = MemoryCoreBuilder(mock_store).enable_keywords()
    assert builder._mode_defaults.enable_keywords is True


def test_disable_keywords(mock_store):
    """Test disable_keywords() disables keyword extraction."""
    builder = MemoryCoreBuilder(mock_store).disable_keywords()
    assert builder._mode_defaults.enable_keywords is False


# --- Retrieval Mode Tests ---


def test_use_retrieval_mode(mock_store):
    """Test use_retrieval_mode() sets the mode."""
    builder = MemoryCoreBuilder(mock_store).use_retrieval_mode("hybrid")
    assert builder._retrieval_config.retrieval_mode == "hybrid"


def test_use_basic_retrieval(mock_store):
    """Test use_basic_retrieval() sets mode to basic."""
    builder = MemoryCoreBuilder(mock_store).use_basic_retrieval()
    assert builder._retrieval_config.retrieval_mode == "basic"


def test_use_hybrid_fusion_defaults(mock_store):
    """Test use_hybrid_fusion() with default parameters."""
    builder = MemoryCoreBuilder(mock_store).use_hybrid_fusion()
    assert builder._retrieval_config.retrieval_mode == "hybrid"
    assert builder._retrieval_config.rrf_k == 60
    assert builder._retrieval_config.enable_rrf_signal_rerank is False
    assert builder._retrieval_config.reranker_backend == "none"


def test_use_hybrid_fusion_custom(mock_store):
    """Test use_hybrid_fusion() with custom parameters."""
    builder = MemoryCoreBuilder(mock_store).use_hybrid_fusion(
        rrf_k=40,
        enable_rrf_signal_rerank=True,
        reranker_backend="flashrank",
        reranker_model="custom-model",
        rerank_candidates=100,
    )
    assert builder._retrieval_config.rrf_k == 40
    assert builder._retrieval_config.enable_rrf_signal_rerank is True
    assert builder._retrieval_config.reranker_backend == "flashrank"
    assert builder._retrieval_config.reranker_model == "custom-model"
    assert builder._retrieval_config.rerank_candidates == 100


# --- BM25 Configuration Tests ---


def test_enable_bm25_defaults(mock_store):
    """Test enable_bm25() with default parameters."""
    builder = MemoryCoreBuilder(mock_store).enable_bm25()
    assert builder._retrieval_config.enable_bm25 is True
    assert abs(builder._retrieval_config.bm25_k1 - 1.2) < 1e-9
    assert abs(builder._retrieval_config.bm25_b - 0.75) < 1e-9


def test_enable_bm25_custom(mock_store):
    """Test enable_bm25() with custom parameters."""
    builder = MemoryCoreBuilder(mock_store).enable_bm25(k1=1.5, b=0.8)
    assert abs(builder._retrieval_config.bm25_k1 - 1.5) < 1e-9
    assert abs(builder._retrieval_config.bm25_b - 0.8) < 1e-9


def test_disable_bm25(mock_store):
    """Test disable_bm25() disables BM25."""
    builder = MemoryCoreBuilder(mock_store).enable_bm25().disable_bm25()
    assert builder._retrieval_config.enable_bm25 is False


# --- PRF Configuration Tests ---


def test_enable_prf_defaults(mock_store):
    """Test enable_prf() with default parameters."""
    builder = MemoryCoreBuilder(mock_store).enable_prf()
    assert builder._retrieval_config.enable_prf is True
    assert builder._retrieval_config.prf_docs == 5
    assert builder._retrieval_config.prf_terms == 8


def test_enable_prf_custom(mock_store):
    """Test enable_prf() with custom parameters."""
    builder = MemoryCoreBuilder(mock_store).enable_prf(docs=10, terms=12)
    assert builder._retrieval_config.prf_docs == 10
    assert builder._retrieval_config.prf_terms == 12


def test_disable_prf(mock_store):
    """Test disable_prf() disables PRF."""
    builder = MemoryCoreBuilder(mock_store).enable_prf().disable_prf()
    assert builder._retrieval_config.enable_prf is False


# --- Defaults Configuration Tests ---


def test_with_defaults(mock_store):
    """Test with_defaults() sets analyze and summarize defaults."""
    builder = MemoryCoreBuilder(mock_store).with_defaults(
        default_analyze=True,
        default_summarize=False,
    )
    assert builder._mode_defaults.default_analyze is True
    assert builder._mode_defaults.default_summarize is False


# --- with_option() Tests ---


def test_with_option_mode(mock_store):
    """Test with_option() for mode."""
    builder = MemoryCoreBuilder(mock_store).with_option("mode", "pro")
    assert builder._mode_defaults.mode == "pro"


def test_with_option_llm(mock_store, mock_llm):
    """Test with_option() for llm."""
    builder = MemoryCoreBuilder(mock_store).with_option("llm", mock_llm)
    assert builder._llm is mock_llm


def test_with_option_embeddings(mock_store, mock_embeddings):
    """Test with_option() for embeddings."""
    builder = MemoryCoreBuilder(mock_store).with_option("embeddings", mock_embeddings)
    assert builder._embeddings is mock_embeddings


def test_with_option_default_importance(mock_store):
    """Test with_option() for default_importance."""
    builder = MemoryCoreBuilder(mock_store).with_option("default_importance", 0.9)
    assert abs(builder._scoring_config.default_importance - 0.9) < 1e-9


def test_with_option_mode_defaults_flags(mock_store):
    """Test with_option() for mode defaults flags."""
    mock_extractor = MagicMock()
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("default_analyze", True)
        .with_option("default_summarize", False)
        .with_option("enable_keywords", True)
        .with_option("keyword_extractor", mock_extractor)
    )
    assert builder._mode_defaults.default_analyze is True
    assert builder._mode_defaults.default_summarize is False
    assert builder._mode_defaults.enable_keywords is True
    assert builder._mode_defaults.keyword_extractor is mock_extractor


def test_with_option_ner_fields(mock_store):
    """Test with_option() for NER config fields."""
    mock_ner = MagicMock()
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("ner", mock_ner)
        .with_option("enable_ner", False)
    )
    assert builder._ner_config.ner is mock_ner
    assert builder._ner_config.enable_ner is False


def test_with_option_scoring_fields(mock_store):
    """Test with_option() for scoring config fields."""

    async def hook(item):
        return item

    scoring = MemoryScoring(importance_weight=0.4, recency_weight=0.4, relevance_weight=0.2)
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("pre_remember_hooks", [hook])
        .with_option("memory_scoring", scoring)
    )
    assert builder._scoring_config.pre_remember_hooks == [hook]
    assert builder._scoring_config.memory_scoring is scoring


def test_with_option_retrieval_mode(mock_store):
    """Test with_option() for retrieval_mode."""
    builder = MemoryCoreBuilder(mock_store).with_option("retrieval_mode", "hybrid")
    assert builder._retrieval_config.retrieval_mode == "hybrid"


def test_with_option_enable_bm25(mock_store):
    """Test with_option() for enable_bm25."""
    builder = MemoryCoreBuilder(mock_store).with_option("enable_bm25", True)
    assert builder._retrieval_config.enable_bm25 is True


def test_with_option_bm25_params(mock_store):
    """Test with_option() for BM25 parameters."""
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("bm25_k1", 1.8)
        .with_option("bm25_b", 0.6)
    )
    assert abs(builder._retrieval_config.bm25_k1 - 1.8) < 1e-9
    assert abs(builder._retrieval_config.bm25_b - 0.6) < 1e-9


def test_with_option_prf_params(mock_store):
    """Test with_option() for PRF parameters."""
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("enable_prf", True)
        .with_option("prf_docs", 7)
        .with_option("prf_terms", 10)
    )
    assert builder._retrieval_config.enable_prf is True
    assert builder._retrieval_config.prf_docs == 7
    assert builder._retrieval_config.prf_terms == 10


def test_with_option_reranker_settings(mock_store):
    """Test with_option() for RRF/reranker settings."""
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("rrf_k", 42)
        .with_option("enable_rrf_signal_rerank", True)
        .with_option("reranker_backend", "flashrank")
        .with_option("reranker_model", "tiny-model")
        .with_option("rerank_candidates", 75)
    )
    assert builder._retrieval_config.rrf_k == 42
    assert builder._retrieval_config.enable_rrf_signal_rerank is True
    assert builder._retrieval_config.reranker_backend == "flashrank"
    assert builder._retrieval_config.reranker_model == "tiny-model"
    assert builder._retrieval_config.rerank_candidates == 75


def test_with_option_async_ingest(mock_store):
    """Test with_option() for ingestion config."""
    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("async_ingest", True)
        .with_option("ingestion_queue_size", 200)
    )
    assert builder._ingestion_config.async_ingest is True
    assert builder._ingestion_config.ingestion_queue_size == 200


def test_with_option_config_objects(mock_store):
    """Test with_option() for config objects."""
    mode_defaults = ModeDefaultsConfig(mode="lite")
    ner_config = NerConfig(enable_ner=False)
    scoring_config = ScoringConfig(default_importance=0.6)
    retrieval_config = RetrievalConfig(enable_bm25=True)
    ingestion_config = IngestionConfig(async_ingest=True)

    builder = (
        MemoryCoreBuilder(mock_store)
        .with_option("mode_defaults", mode_defaults)
        .with_option("ner_config", ner_config)
        .with_option("scoring_config", scoring_config)
        .with_option("retrieval_config", retrieval_config)
        .with_option("ingestion_config", ingestion_config)
    )

    assert builder._mode_defaults is mode_defaults
    assert builder._ner_config is ner_config
    assert builder._scoring_config is scoring_config
    assert builder._retrieval_config is retrieval_config
    assert builder._ingestion_config is ingestion_config


def test_with_option_unknown_raises(mock_store):
    """Test with_option() raises for unknown options."""
    with pytest.raises(ValueError, match="Unknown MemoryCore option"):
        MemoryCoreBuilder(mock_store).with_option("unknown_option", "value")


# --- Build Integration Tests ---


def test_build_passes_all_configs(mock_store, mock_llm, mock_embeddings):
    """Test that build() creates MemoryCore with all configurations."""
    mock_ner = MagicMock()

    core = (
        MemoryCoreBuilder(mock_store)
        .with_llm(mock_llm)
        .with_embeddings(mock_embeddings)
        .with_mode("pro")
        .with_ner(mock_ner, enable=False)  # Disable to avoid loading spacy
        .disable_keywords()  # Disable to avoid loading spacy
        .with_default_importance(0.7)
        .enable_bm25()
        .use_hybrid_fusion()
        .build()
    )

    # Verify core attributes
    assert core.store is mock_store
    assert core.embedder is mock_embeddings  # MemoryCore stores embeddings as 'embedder'
    assert abs(core.default_importance - 0.7) < 1e-9
    assert core.enable_bm25 is True
    assert core.retrieval_mode == "hybrid"
