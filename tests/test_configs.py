"""Tests for configuration module."""

from unittest.mock import MagicMock, patch

import pytest

from mnemotree.configs import (
    ConfiguredMemorySystem,
    MemorySystemConfig,
    coding_copilot_config,
    correctness_first_config,
    customer_support_config,
    helpfulness_first_config,
    high_performance_config,
    personal_assistant_config,
    research_assistant_config,
)
from mnemotree.core.hybrid_retrieval import FusionStrategy
from mnemotree.experimental import ConsolidationConfig, DecayParameters, WritePolicy


class TestMemorySystemConfig:
    """Tests for MemorySystemConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MemorySystemConfig()
        assert config.llm is None
        assert config.embeddings is None
        assert config.use_hybrid_retrieval is True
        assert config.use_reranking is True
        assert config.fusion_strategy == FusionStrategy.RRF
        assert config.enable_consolidation is True
        assert config.enable_truth_maintenance is True
        assert config.enable_adaptive_decay is True
        assert config.enable_write_gate is True

    def test_custom_values(self):
        """Test configuration with custom values."""
        mock_llm = MagicMock()
        mock_embeddings = MagicMock()
        decay_params = DecayParameters.default()
        write_policy = WritePolicy.strict()

        config = MemorySystemConfig(
            llm=mock_llm,
            embeddings=mock_embeddings,
            use_hybrid_retrieval=False,
            use_reranking=False,
            fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            enable_consolidation=False,
            staleness_threshold_days=30,
            decay_params=decay_params,
            write_policy=write_policy,
        )

        assert config.llm is mock_llm
        assert config.embeddings is mock_embeddings
        assert config.use_hybrid_retrieval is False
        assert config.fusion_strategy == FusionStrategy.WEIGHTED_SUM
        assert config.staleness_threshold_days == 30


class TestConfiguredMemorySystem:
    """Tests for ConfiguredMemorySystem dataclass."""

    def test_creation_with_memory_core_only(self):
        """ConfiguredMemorySystem can be created with just memory_core."""
        mock_core = MagicMock()
        system = ConfiguredMemorySystem(memory_core=mock_core)

        assert system.memory_core is mock_core
        assert system.retriever is None
        assert system.consolidator is None
        assert system.claims_registry is None
        assert system.adaptive_system is None
        assert system.write_gate is None

    def test_creation_with_all_components(self):
        """ConfiguredMemorySystem can have all components."""
        mock_core = MagicMock()
        mock_retriever = MagicMock()
        mock_consolidator = MagicMock()
        mock_registry = MagicMock()
        mock_adaptive = MagicMock()
        mock_gate = MagicMock()

        system = ConfiguredMemorySystem(
            memory_core=mock_core,
            retriever=mock_retriever,
            consolidator=mock_consolidator,
            claims_registry=mock_registry,
            adaptive_system=mock_adaptive,
            write_gate=mock_gate,
        )

        assert system.retriever is mock_retriever
        assert system.consolidator is mock_consolidator
        assert system.claims_registry is mock_registry
        assert system.adaptive_system is mock_adaptive
        assert system.write_gate is mock_gate


class TestPreConfiguredSetups:
    """Tests for pre-configured setup functions."""

    def test_personal_assistant_config(self):
        """personal_assistant_config returns valid configuration."""
        config = personal_assistant_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.use_hybrid_retrieval is True
        assert config.use_reranking is True
        assert config.enable_consolidation is True
        assert config.enable_spaced_repetition is True
        assert config.consolidation_config is not None
        assert config.decay_params is not None
        assert config.write_policy is not None

    def test_coding_copilot_config(self):
        """coding_copilot_config returns speed-optimized configuration."""
        config = coding_copilot_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.use_reranking is False  # Speed optimization
        assert config.fusion_strategy == FusionStrategy.WEIGHTED_SUM
        assert config.enable_spaced_repetition is False

    def test_customer_support_config(self):
        """customer_support_config returns precision-focused configuration."""
        config = customer_support_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.use_reranking is True
        assert config.enable_truth_maintenance is True
        # Check PII blocking in write policy
        assert config.write_policy is not None
        assert config.write_policy.block_pii is True

    def test_research_assistant_config(self):
        """research_assistant_config returns quality-focused configuration."""
        config = research_assistant_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.enable_truth_maintenance is True
        assert config.staleness_threshold_days == 180
        # High novelty threshold
        assert config.write_policy is not None
        assert config.write_policy.min_novelty_score >= 0.7

    def test_high_performance_config(self):
        """high_performance_config returns speed-optimized configuration."""
        config = high_performance_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.use_hybrid_retrieval is False
        assert config.use_reranking is False
        assert config.enable_consolidation is False
        assert config.enable_truth_maintenance is False

    def test_correctness_first_config(self):
        """correctness_first_config returns strict configuration."""
        config = correctness_first_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.enable_truth_maintenance is True
        assert config.staleness_threshold_days == 30  # Aggressive
        # Conservative consolidation
        assert config.consolidation_config is not None
        assert config.consolidation_config.similarity_threshold >= 0.85

    def test_helpfulness_first_config(self):
        """helpfulness_first_config returns permissive configuration."""
        config = helpfulness_first_config()

        assert isinstance(config, MemorySystemConfig)
        assert config.use_reranking is False
        assert config.fusion_strategy == FusionStrategy.MAX_SCORE
        # Aggressive consolidation to manage volume
        assert config.consolidation_config is not None
        assert config.consolidation_config.min_cluster_size == 2


class TestBuildMemorySystem:
    """Tests for MemorySystemConfig.build_memory_system()."""

    @patch("mnemotree.configs.ChatOpenAI")
    @patch("mnemotree.configs.OpenAIEmbeddings")
    @patch("mnemotree.configs.MemoryCore")
    def test_build_creates_memory_core(self, mock_core_cls, mock_embeddings_cls, mock_llm_cls):
        """build_memory_system creates MemoryCore with provided store."""
        mock_store = MagicMock()
        mock_core_cls.return_value = MagicMock()

        config = MemorySystemConfig(
            enable_consolidation=False,
            enable_truth_maintenance=False,
            enable_adaptive_decay=False,
            enable_write_gate=False,
        )
        system = config.build_memory_system(mock_store)

        assert isinstance(system, ConfiguredMemorySystem)
        mock_core_cls.assert_called_once()
        call_kwargs = mock_core_cls.call_args[1]
        assert call_kwargs["store"] is mock_store

    @patch("mnemotree.configs.ChatOpenAI")
    @patch("mnemotree.configs.OpenAIEmbeddings")
    @patch("mnemotree.configs.MemoryCore")
    @patch("mnemotree.configs.HybridRetriever")
    def test_build_creates_hybrid_retriever(
        self, mock_retriever_cls, mock_core_cls, mock_embeddings_cls, mock_llm_cls
    ):
        """build_memory_system creates HybridRetriever when enabled."""
        mock_store = MagicMock()
        mock_core_cls.return_value = MagicMock()
        mock_retriever_cls.return_value = MagicMock()

        config = MemorySystemConfig(
            use_hybrid_retrieval=True,
            enable_consolidation=False,
            enable_truth_maintenance=False,
            enable_adaptive_decay=False,
            enable_write_gate=False,
        )
        system = config.build_memory_system(mock_store)

        mock_retriever_cls.assert_called_once()
        assert system.retriever is not None

    @patch("mnemotree.configs.ChatOpenAI")
    @patch("mnemotree.configs.OpenAIEmbeddings")
    @patch("mnemotree.configs.MemoryCore")
    def test_build_skips_hybrid_retriever_when_disabled(
        self, mock_core_cls, mock_embeddings_cls, mock_llm_cls
    ):
        """build_memory_system skips HybridRetriever when disabled."""
        mock_store = MagicMock()
        mock_core_cls.return_value = MagicMock()

        config = MemorySystemConfig(
            use_hybrid_retrieval=False,
            enable_consolidation=False,
            enable_truth_maintenance=False,
            enable_adaptive_decay=False,
            enable_write_gate=False,
        )
        system = config.build_memory_system(mock_store)

        assert system.retriever is None

    @patch("mnemotree.configs.ChatOpenAI")
    @patch("mnemotree.configs.OpenAIEmbeddings")
    @patch("mnemotree.configs.MemoryCore")
    def test_build_uses_provided_llm_and_embeddings(
        self, mock_core_cls, mock_embeddings_cls, mock_llm_cls
    ):
        """build_memory_system uses provided LLM and embeddings."""
        mock_store = MagicMock()
        mock_llm = MagicMock()
        mock_embeddings = MagicMock()
        mock_core_cls.return_value = MagicMock()

        config = MemorySystemConfig(
            llm=mock_llm,
            embeddings=mock_embeddings,
            enable_consolidation=False,
            enable_truth_maintenance=False,
            enable_adaptive_decay=False,
            enable_write_gate=False,
        )
        system = config.build_memory_system(mock_store)

        # Should not create new instances
        mock_llm_cls.assert_not_called()
        mock_embeddings_cls.assert_not_called()

        # Should use provided instances
        call_kwargs = mock_core_cls.call_args[1]
        assert call_kwargs["llm"] is mock_llm
        assert call_kwargs["embeddings"] is mock_embeddings


class TestDecayParameters:
    """Tests for DecayParameters defaults."""

    def test_default_factory(self):
        """DecayParameters.default() creates valid parameters."""
        params = DecayParameters.default()
        assert params is not None
        assert hasattr(params, "base_decay_rates")


class TestWritePolicy:
    """Tests for WritePolicy presets."""

    def test_balanced_policy(self):
        """WritePolicy.balanced() creates moderate settings."""
        policy = WritePolicy.balanced()
        assert policy is not None
        assert 0 < policy.min_novelty_score < 1

    def test_strict_policy(self):
        """WritePolicy.strict() creates high thresholds."""
        policy = WritePolicy.strict()
        assert policy is not None
        assert policy.min_novelty_score >= 0.6
        assert policy.min_confidence >= 0.7

    def test_permissive_policy(self):
        """WritePolicy.permissive() creates low thresholds."""
        policy = WritePolicy.permissive()
        assert policy is not None
        assert policy.min_novelty_score <= 0.3
