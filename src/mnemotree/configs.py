"""
Configuration examples for different use cases.

This module provides pre-configured setups optimized for specific scenarios.
"""

import os
from dataclasses import dataclass

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mnemotree.core import MemoryCore
from mnemotree.core.models import MemoryType
from mnemotree.rerankers import CrossEncoderReranker, NoOpReranker
from mnemotree.core.hybrid_retrieval import FusionStrategy, HybridRetriever
from mnemotree.experimental import (
    AdaptiveImportanceSystem,
    ClaimsRegistry,
    ConsolidationConfig,
    ContextAwareWriteGate,
    DecayParameters,
    MemoryConsolidator,
    WritePolicy,
)
from mnemotree.store.protocols import MemoryCRUDStore


@dataclass
class MemorySystemConfig:
    """Complete configuration for a memory system."""

    # Core
    llm: BaseLanguageModel | None = None
    embeddings: Embeddings | None = None

    # Retrieval
    use_hybrid_retrieval: bool = True
    use_reranking: bool = True
    fusion_strategy: FusionStrategy = FusionStrategy.RRF

    # Consolidation
    enable_consolidation: bool = True
    consolidation_config: ConsolidationConfig | None = None

    # Truth maintenance
    enable_truth_maintenance: bool = True
    staleness_threshold_days: int = 90

    # Adaptive decay
    enable_adaptive_decay: bool = True
    enable_spaced_repetition: bool = True
    decay_params: DecayParameters | None = None

    # Write gating
    enable_write_gate: bool = True
    write_policy: WritePolicy | None = None

    def build_memory_system(self, store: MemoryCRUDStore) -> "ConfiguredMemorySystem":
        """Build a fully configured memory system."""

        # Initialize LLM and embeddings if not provided.
        # Prefer explicit constructor args; otherwise fall back to env-configured defaults.
        llm = self.llm or ChatOpenAI(
            model=os.getenv("MNEMOTREE_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4",
            temperature=float(os.getenv("MNEMOTREE_LLM_TEMPERATURE", "0.7")),
        )

        embedding_model = os.getenv("MNEMOTREE_EMBEDDING_MODEL") or os.getenv(
            "OPENAI_EMBEDDING_MODEL"
        )
        embeddings = self.embeddings or (
            OpenAIEmbeddings(model=embedding_model) if embedding_model else OpenAIEmbeddings()
        )

        # Core memory
        memory_core = MemoryCore(
            store=store,
            llm=llm,
            embeddings=embeddings,
        )

        # Hybrid retrieval
        retriever = None
        if self.use_hybrid_retrieval:
            retriever = HybridRetriever(
                fusion_strategy=self.fusion_strategy,
                reranker=CrossEncoderReranker() if self.use_reranking else NoOpReranker(),
            )

        # Consolidation
        consolidator = None
        if self.enable_consolidation:
            config = self.consolidation_config or ConsolidationConfig()
            consolidator = MemoryConsolidator(llm=llm, config=config)

        # Truth maintenance
        claims_registry = None
        if self.enable_truth_maintenance:
            claims_registry = ClaimsRegistry(
                llm=llm,
                staleness_threshold_days=self.staleness_threshold_days,
            )

        # Adaptive decay
        adaptive_system = None
        if self.enable_adaptive_decay:
            params = self.decay_params or DecayParameters.default()
            adaptive_system = AdaptiveImportanceSystem(
                decay_params=params,
                enable_spaced_repetition=self.enable_spaced_repetition,
            )

        # Write gating
        write_gate = None
        if self.enable_write_gate:
            policy = self.write_policy or WritePolicy.balanced()
            write_gate = ContextAwareWriteGate(policy=policy)

        return ConfiguredMemorySystem(
            memory_core=memory_core,
            retriever=retriever,
            consolidator=consolidator,
            claims_registry=claims_registry,
            adaptive_system=adaptive_system,
            write_gate=write_gate,
        )


@dataclass
class ConfiguredMemorySystem:
    """A fully configured memory system with all components."""

    memory_core: MemoryCore
    retriever: HybridRetriever | None = None
    consolidator: MemoryConsolidator | None = None
    claims_registry: ClaimsRegistry | None = None
    adaptive_system: AdaptiveImportanceSystem | None = None
    write_gate: ContextAwareWriteGate | None = None


# ============================================================================
# PRE-CONFIGURED SETUPS
# ============================================================================


def personal_assistant_config() -> MemorySystemConfig:
    """
    Configuration optimized for personal assistant use case.

    Characteristics:
    - Balanced quality filtering
    - Moderate consolidation
    - Emphasis on user preferences and routines
    - Spaced repetition for important information
    """
    return MemorySystemConfig(
        # Retrieval
        use_hybrid_retrieval=True,
        use_reranking=True,
        fusion_strategy=FusionStrategy.RRF,
        # Consolidation - moderate frequency
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            min_cluster_size=3,
            similarity_threshold=0.7,
            importance_threshold=0.2,
            age_threshold_days=30,
            preserve_high_importance=0.8,
        ),
        # Truth maintenance - track preferences
        enable_truth_maintenance=True,
        staleness_threshold_days=60,
        # Adaptive decay - preserve user preferences
        enable_adaptive_decay=True,
        enable_spaced_repetition=True,
        decay_params=DecayParameters(
            base_decay_rates={
                MemoryType.EPISODIC: 0.02,
                MemoryType.SEMANTIC: 0.003,  # Slower for facts
                MemoryType.PROSPECTIVE: 0.01,  # Remember TODOs
            },
            access_frequency_weight=0.4,
            recency_weight=0.3,
            novelty_weight=0.3,
        ),
        # Write gating - balanced
        enable_write_gate=True,
        write_policy=WritePolicy(
            min_novelty_score=0.4,
            min_confidence=0.6,
            min_content_length=10,
            require_meaningful_content=True,
            max_memories_per_hour=100,
        ),
    )


def coding_copilot_config() -> MemorySystemConfig:
    """
    Configuration optimized for coding copilot use case.

    Characteristics:
    - Fast retrieval (no reranking)
    - Aggressive consolidation for patterns
    - High preservation of important code snippets
    - Permissive storage for context
    """
    return MemorySystemConfig(
        # Retrieval - speed over precision
        use_hybrid_retrieval=True,
        use_reranking=False,  # Too slow for IDE
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
        # Consolidation - aggressive for code patterns
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            min_cluster_size=2,
            similarity_threshold=0.8,
            importance_threshold=0.15,
            age_threshold_days=60,
            preserve_high_importance=0.9,
        ),
        # Truth maintenance - track API changes
        enable_truth_maintenance=True,
        staleness_threshold_days=30,  # Code changes fast
        # Adaptive decay - slow for established patterns
        enable_adaptive_decay=True,
        enable_spaced_repetition=False,  # Not needed for code
        decay_params=DecayParameters(
            base_decay_rates={
                MemoryType.EPISODIC: 0.03,
                MemoryType.SEMANTIC: 0.001,  # Very slow for code knowledge
                MemoryType.PROCEDURAL: 0.0005,  # Almost never forget patterns
            },
        ),
        # Write gating - permissive
        enable_write_gate=True,
        write_policy=WritePolicy.permissive(),
    )


def customer_support_config() -> MemorySystemConfig:
    """
    Configuration optimized for customer support use case.

    Characteristics:
    - High precision retrieval
    - Strict quality filtering
    - Strong truth maintenance
    - PII detection enabled
    """
    return MemorySystemConfig(
        # Retrieval - precision critical
        use_hybrid_retrieval=True,
        use_reranking=True,
        fusion_strategy=FusionStrategy.RRF,
        # Consolidation - preserve customer history
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            min_cluster_size=4,
            similarity_threshold=0.75,
            importance_threshold=0.3,
            age_threshold_days=90,
            preserve_high_importance=0.85,
        ),
        # Truth maintenance - critical for policies
        enable_truth_maintenance=True,
        staleness_threshold_days=45,
        # Adaptive decay - moderate
        enable_adaptive_decay=True,
        enable_spaced_repetition=True,
        # Write gating - strict + PII blocking
        enable_write_gate=True,
        write_policy=WritePolicy(
            min_novelty_score=0.5,
            min_confidence=0.7,
            min_content_length=15,
            require_meaningful_content=True,
            block_pii=True,  # Critical for privacy
            max_memories_per_hour=200,
        ),
    )


def research_assistant_config() -> MemorySystemConfig:
    """
    Configuration optimized for research assistant use case.

    Characteristics:
    - High quality requirements
    - Strong novelty emphasis
    - Conservative consolidation
    - Detailed provenance tracking
    """
    return MemorySystemConfig(
        # Retrieval - quality matters
        use_hybrid_retrieval=True,
        use_reranking=True,
        fusion_strategy=FusionStrategy.RRF,
        # Consolidation - conservative
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            min_cluster_size=5,
            similarity_threshold=0.8,
            importance_threshold=0.1,
            age_threshold_days=120,
            preserve_high_importance=0.9,
        ),
        # Truth maintenance - track sources and conflicts
        enable_truth_maintenance=True,
        staleness_threshold_days=180,
        # Adaptive decay - preserve findings
        enable_adaptive_decay=True,
        enable_spaced_repetition=True,
        decay_params=DecayParameters(
            base_decay_rates={
                MemoryType.EPISODIC: 0.015,
                MemoryType.SEMANTIC: 0.002,
                MemoryType.AUTOBIOGRAPHICAL: 0.001,
            },
            novelty_weight=0.4,  # Emphasize novelty
        ),
        # Write gating - high novelty threshold
        enable_write_gate=True,
        write_policy=WritePolicy(
            min_novelty_score=0.7,
            min_confidence=0.7,
            min_content_length=20,
            require_meaningful_content=True,
            allow_redundant=False,
        ),
    )


def high_performance_config() -> MemorySystemConfig:
    """
    Configuration optimized for speed over quality.

    Characteristics:
    - Minimal overhead
    - Fast retrieval
    - Disabled expensive features
    - Suitable for high-throughput scenarios
    """
    return MemorySystemConfig(
        # Retrieval - basic only
        use_hybrid_retrieval=False,
        use_reranking=False,
        # Consolidation - disabled (run offline)
        enable_consolidation=False,
        # Truth maintenance - disabled
        enable_truth_maintenance=False,
        # Adaptive decay - basic only
        enable_adaptive_decay=True,
        enable_spaced_repetition=False,
        # Write gating - minimal checks
        enable_write_gate=True,
        write_policy=WritePolicy(
            min_novelty_score=0.2,
            min_confidence=0.3,
            min_content_length=5,
            require_meaningful_content=False,
        ),
    )


def correctness_first_config() -> MemorySystemConfig:
    """
    Configuration prioritizing correctness over recall.

    Characteristics:
    - Strict quality requirements
    - Aggressive truth maintenance
    - High confidence thresholds
    - Conservative consolidation
    """
    return MemorySystemConfig(
        # Retrieval - precision over recall
        use_hybrid_retrieval=True,
        use_reranking=True,
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
        # Consolidation - very conservative
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            min_cluster_size=5,
            similarity_threshold=0.85,
            importance_threshold=0.4,
            preserve_high_importance=0.95,
        ),
        # Truth maintenance - aggressive
        enable_truth_maintenance=True,
        staleness_threshold_days=30,
        # Adaptive decay - minimal
        enable_adaptive_decay=True,
        enable_spaced_repetition=True,
        # Write gating - very strict
        enable_write_gate=True,
        write_policy=WritePolicy.strict(),
    )


def helpfulness_first_config() -> MemorySystemConfig:
    """
    Configuration prioritizing recall over precision.

    Characteristics:
    - Permissive storage
    - Broad retrieval
    - Minimal filtering
    - Aggressive consolidation to manage growth
    """
    return MemorySystemConfig(
        # Retrieval - broad recall
        use_hybrid_retrieval=True,
        use_reranking=False,
        fusion_strategy=FusionStrategy.MAX_SCORE,
        # Consolidation - aggressive to manage volume
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            min_cluster_size=2,
            similarity_threshold=0.6,
            importance_threshold=0.1,
        ),
        # Truth maintenance - basic
        enable_truth_maintenance=True,
        staleness_threshold_days=120,
        # Adaptive decay - moderate
        enable_adaptive_decay=True,
        enable_spaced_repetition=False,
        # Write gating - permissive
        enable_write_gate=True,
        write_policy=WritePolicy.permissive(),
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Personal Assistant

from mnemotree.store.chromadb_store import ChromaDBStore
from mnemotree.configs import personal_assistant_config

config = personal_assistant_config()
store = ChromaDBStore(collection_name="my_assistant")
system = config.build_memory_system(store)

# Use the configured system
memory = system.memory_core
await memory.remember("I prefer dark mode")


# Example 2: Coding Copilot (fast)

from mnemotree.configs import coding_copilot_config

config = coding_copilot_config()
store = ChromaDBStore(collection_name="code_memory")
system = config.build_memory_system(store)

# Fast retrieval without reranking
results = await system.memory_core.recall("error handling pattern")


# Example 3: Custom Configuration

config = MemorySystemConfig(
    use_hybrid_retrieval=True,
    use_reranking=True,
    enable_consolidation=False,  # Disabled
    write_policy=WritePolicy(
        min_novelty_score=0.6,
        min_confidence=0.8,
    ),
)

system = config.build_memory_system(store)


# Example 4: Access Individual Components

system = personal_assistant_config().build_memory_system(store)

# Use write gate
from mnemotree.core import WriteDecision

result = await system.write_gate.evaluate(candidate_memory)
if result.decision == WriteDecision.ACCEPT:
    await system.memory_core.remember(candidate_memory.content)

# Run consolidation
if system.consolidator:
    all_memories = await store.get_all_memories()
    consolidation_result = await system.consolidator.consolidate(all_memories)

# Check for conflicts
if system.claims_registry:
    conflicts = system.claims_registry.get_active_conflicts()
"""
