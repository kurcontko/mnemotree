"""Smoke tests: verify core imports and basic construction without optional deps."""

# ruff: noqa: F401, I001

from __future__ import annotations


def test_core_imports():
    """All public API symbols should be importable."""
    from mnemotree import (
        ChainOfRetrieval,
        CoRAGConfig,
        ConfigurationError,
        DependencyError,
        GapAnalyzer,
        HeuristicGapAnalyzer,
        InvalidQueryError,
        LinkType,
        MemoryCore,
        MemoryCoreBuilder,
        MemoryLink,
        MemoryMode,
        MemoryNotFoundError,
        MnemotreeError,
        PPRConfig,
        PPRGraphRetriever,
        RecallFilters,
        RecallOptions,
        RememberOptions,
        SCMRAGRetriever,
        SerializationError,
        StoreError,
        TypedPathQuery,
        TypedPathRetriever,
    )

    assert MemoryCore is not None
    assert MemoryCoreBuilder is not None


def test_store_protocol_imports():
    """Store protocols should be importable without any backend installed."""
    from mnemotree.store import (
        BaseMemoryStore,
        MemoryCRUDStore,
        SupportsEntityQuery,
        SupportsMemoryListing,
        SupportsMetadataUpdate,
        SupportsStructuredQuery,
        SupportsVectorSearch,
    )

    assert MemoryCRUDStore is not None


def test_core_submodule_imports():
    """Core submodule re-exports should work."""
    from mnemotree.core import (
        FusionStrategy,
        HybridRetriever,
        IngestionConfig,
        MemoryItem,
        MemoryType,
        ModeDefaultsConfig,
        NerConfig,
        RetrievalConfig,
        RetrieverFactory,
        ScoringConfig,
    )

    assert MemoryItem is not None
    assert MemoryType.SEMANTIC is not None


def test_experimental_imports():
    """Experimental module should be importable."""
    from mnemotree.experimental import (
        ConsolidationConfig,
        ContextAwareWriteGate,
        DecayParameters,
        MemoryConsolidator,
        WritePolicy,
    )

    assert ConsolidationConfig is not None


def test_memory_item_creation():
    """MemoryItem can be created with minimal fields."""
    from mnemotree.core.models import MemoryItem, MemoryType

    item = MemoryItem(
        content="test memory",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
    )
    assert item.content == "test memory"
    assert item.memory_id  # auto-generated


def test_builder_exists():
    """MemoryCoreBuilder.default() should be callable."""
    from mnemotree import MemoryCoreBuilder

    # Just verify the class and classmethod exist
    assert hasattr(MemoryCoreBuilder, "default")
    assert hasattr(MemoryCoreBuilder, "lite")
    assert hasattr(MemoryCoreBuilder, "pro")


def test_configs_presets():
    """Pre-configured setups should return valid configs."""
    from mnemotree.configs import (
        coding_copilot_config,
        correctness_first_config,
        customer_support_config,
        helpfulness_first_config,
        high_performance_config,
        personal_assistant_config,
        research_assistant_config,
    )

    for factory in [
        personal_assistant_config,
        coding_copilot_config,
        customer_support_config,
        research_assistant_config,
        high_performance_config,
        correctness_first_config,
        helpfulness_first_config,
    ]:
        config = factory()
        assert config is not None
