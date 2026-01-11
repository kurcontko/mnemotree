"""
Memory consolidation and distillation system.

Implements "sleep" cycles that:
1. Cluster episodic memories based on similarity
2. Summarize clusters into semantic memories
3. Deprecate low-signal instances while preserving provenance
4. Maintain lineage tracking for auditability
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import numpy as np
from langchain_core.language_models.base import BaseLanguageModel
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from ..analysis.clustering import MemoryClusterer
from ..core.models import MemoryItem, MemoryType, coerce_datetime


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""

    # Clustering parameters
    min_cluster_size: int = 3
    clustering_method: str = "dbscan"  # "dbscan" or "agglomerative"
    similarity_threshold: float = 0.7

    # Deprecation criteria
    importance_threshold: float = 0.2  # Below this, consider for deprecation
    access_threshold: int = 0  # Memories with <= this many accesses
    age_threshold_days: int = 30  # Older than this

    # Consolidation behavior
    preserve_high_importance: float = 0.7  # Always keep memories above this
    max_cluster_summary_length: int = 500

    # Safety settings
    dry_run: bool = False  # If True, don't actually deprecate
    preserve_provenance: bool = True  # Keep metadata links


@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle."""

    # Statistics
    total_memories_processed: int
    clusters_formed: int
    semantic_memories_created: int
    memories_deprecated: int

    # Details
    created_semantic_ids: list[str]
    deprecated_memory_ids: list[str]
    cluster_summaries: list[dict[str, Any]]

    # Timing
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


class MemoryConsolidator:
    """
    Memory consolidation engine for sleep-like processing.

    Mimics biological memory consolidation where:
    - Similar episodic memories are grouped
    - Clusters are abstracted into semantic knowledge
    - Low-value memories are pruned
    - Important details are preserved
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        clusterer: MemoryClusterer | None = None,
        config: ConsolidationConfig | None = None,
    ):
        """
        Initialize consolidator.

        Args:
            llm: Language model for summarization
            clusterer: Memory clustering component
            config: Consolidation configuration
        """
        self.llm = llm
        self.clusterer = clusterer or MemoryClusterer()
        self.config = config or ConsolidationConfig()

    async def consolidate(
        self,
        memories: list[MemoryItem],
        user_id: str | None = None,
    ) -> ConsolidationResult:
        """
        Run a consolidation cycle on a set of memories.

        Args:
            memories: Episodic memories to consolidate
            user_id: Optional user context for personalization

        Returns:
            ConsolidationResult with statistics and created artifacts
        """
        started_at = datetime.now(timezone.utc)

        # Filter to episodic memories suitable for consolidation
        episodic_memories = [m for m in memories if m.memory_type == MemoryType.EPISODIC]

        if len(episodic_memories) < self.config.min_cluster_size:
            return self._empty_result(started_at)

        # Step 1: Cluster similar episodic memories
        clusters = await self._cluster_memories(episodic_memories)

        # Step 2: Generate semantic memories from clusters
        semantic_memories, cluster_summaries = await self._generate_semantic_memories(
            clusters, user_id
        )

        # Step 3: Identify memories to deprecate
        deprecated_ids = self._identify_deprecated_memories(episodic_memories, clusters)

        completed_at = datetime.now(timezone.utc)
        duration = (completed_at - started_at).total_seconds()

        return ConsolidationResult(
            total_memories_processed=len(episodic_memories),
            clusters_formed=len(clusters),
            semantic_memories_created=len(semantic_memories),
            memories_deprecated=len(deprecated_ids),
            created_semantic_ids=[m.memory_id for m in semantic_memories],
            deprecated_memory_ids=deprecated_ids,
            cluster_summaries=cluster_summaries,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    async def _cluster_memories(self, memories: list[MemoryItem]) -> list[list[MemoryItem]]:
        """
        Cluster memories by semantic similarity.

        Args:
            memories: Memories to cluster

        Returns:
            List of memory clusters
        """
        # Extract embeddings
        embeddings = []
        for memory in memories:
            if memory.embedding:
                embeddings.append(memory.embedding)
            else:
                embeddings.append([0.0] * 768)  # Placeholder

        embeddings_array = np.array(embeddings)

        # Cluster based on configuration
        if self.config.clustering_method == "dbscan":
            clusterer = DBSCAN(
                eps=1 - self.config.similarity_threshold,
                min_samples=self.config.min_cluster_size,
                metric="cosine",
            )
        else:  # agglomerative
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.config.similarity_threshold,
                linkage="average",
            )

        labels = clusterer.fit_predict(embeddings_array)

        # Group memories by cluster label
        clusters: dict[int, list[MemoryItem]] = {}
        for memory, label in zip(memories, labels, strict=True):
            if label == -1:  # Noise points in DBSCAN
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memory)

        # Filter out small clusters
        return [
            cluster for cluster in clusters.values() if len(cluster) >= self.config.min_cluster_size
        ]

    async def _generate_semantic_memories(
        self,
        clusters: list[list[MemoryItem]],
        user_id: str | None = None,
    ) -> tuple[list[MemoryItem], list[dict[str, Any]]]:
        """
        Generate semantic memories from episodic clusters.

        Args:
            clusters: Clustered episodic memories
            user_id: Optional user context

        Returns:
            Tuple of (created semantic memories, cluster summaries)
        """
        semantic_memories = []
        cluster_summaries = []

        for cluster in clusters:
            # Create summary of cluster
            summary = await self._summarize_cluster(cluster)

            # Extract common themes, entities, tags
            common_tags = self._extract_common_tags(cluster)
            common_entities = self._extract_common_entities(cluster)

            # Calculate aggregate metrics
            avg_importance = np.mean([m.importance for m in cluster])
            total_accesses = sum(m.access_count for m in cluster)

            # Create semantic memory
            semantic_memory = MemoryItem(
                memory_id=str(uuid4()),
                content=summary,
                summary=summary[:200] + "..." if len(summary) > 200 else summary,
                memory_type=MemoryType.SEMANTIC,
                importance=min(1.0, avg_importance + 0.1),  # Boost consolidated memories
                tags=common_tags,
                entities=common_entities,
                access_count=total_accesses,
                confidence=0.8,  # Lower confidence for synthesized memories
                user_id=user_id,
                metadata={
                    "consolidated": True,
                    "source_memories": [m.memory_id for m in cluster],
                    "cluster_size": len(cluster),
                    "consolidation_date": str(datetime.now(timezone.utc)),
                },
            )

            semantic_memories.append(semantic_memory)

            cluster_summaries.append(
                {
                    "semantic_memory_id": semantic_memory.memory_id,
                    "cluster_size": len(cluster),
                    "source_memory_ids": [m.memory_id for m in cluster],
                    "summary": summary,
                    "common_tags": common_tags,
                }
            )

        return semantic_memories, cluster_summaries

    async def _summarize_cluster(self, cluster: list[MemoryItem]) -> str:
        """
        Generate a summary of a memory cluster.

        Args:
            cluster: List of related memories

        Returns:
            Summary text
        """
        # Combine memory contents
        contents = [m.content for m in cluster]
        combined = "\n\n".join(contents)

        # Prompt for summarization
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            """You are a memory consolidation system. Given a cluster of related episodic memories,
create a concise semantic summary that captures the key facts, patterns, and insights.

Episodic Memories:
{memories}

Generate a clear, factual summary (max {max_length} chars) that:
1. Identifies common themes and patterns
2. Extracts key facts and insights
3. Notes important relationships or sequences
4. Maintains factual accuracy

Summary:"""
        )

        chain = prompt | self.llm
        result = await chain.ainvoke(
            {
                "memories": combined[:4000],  # Limit input length
                "max_length": self.config.max_cluster_summary_length,
            }
        )

        if hasattr(result, "content"):
            return result.content.strip()
        return str(result).strip()

    def _extract_common_tags(self, cluster: list[MemoryItem]) -> list[str]:
        """Extract tags that appear in multiple cluster memories."""
        tag_counts: dict[str, int] = {}

        for memory in cluster:
            if memory.tags:
                for tag in memory.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Keep tags that appear in at least 30% of cluster
        threshold = len(cluster) * 0.3
        common_tags = [tag for tag, count in tag_counts.items() if count >= threshold]

        return common_tags

    def _extract_common_entities(self, cluster: list[MemoryItem]) -> dict[str, str]:
        """Extract entities that appear in multiple cluster memories."""
        entity_counts: dict[str, tuple[str, int]] = {}

        for memory in cluster:
            if memory.entities:
                for entity_text, entity_type in memory.entities.items():
                    key = entity_text.lower()
                    if key in entity_counts:
                        entity_counts[key] = (entity_type, entity_counts[key][1] + 1)
                    else:
                        entity_counts[key] = (entity_type, 1)

        # Keep entities that appear in at least 2 memories
        threshold = min(2, len(cluster))
        common_entities = {
            entity: entity_type
            for entity, (entity_type, count) in entity_counts.items()
            if count >= threshold
        }

        return common_entities

    def _identify_deprecated_memories(
        self,
        all_memories: list[MemoryItem],
        clusters: list[list[MemoryItem]],
    ) -> list[str]:
        """
        Identify memories that should be deprecated.

        Criteria:
        - Low importance AND low access count AND old
        - Part of a consolidated cluster (redundant)
        - Never high-importance memories (safety check)

        Args:
            all_memories: All episodic memories
            clusters: Formed clusters

        Returns:
            List of memory IDs to deprecate
        """
        deprecated_ids = []

        # Create set of memory IDs in clusters
        clustered_ids = {m.memory_id for cluster in clusters for m in cluster}

        # Calculate age threshold
        age_threshold = datetime.now(timezone.utc) - timedelta(days=self.config.age_threshold_days)

        for memory in all_memories:
            # Never deprecate high-importance memories
            if memory.importance >= self.config.preserve_high_importance:
                continue

            # Parse timestamp
            mem_time = coerce_datetime(memory.timestamp, default=None)
            if mem_time is None:
                continue

            # Check deprecation criteria
            is_old = mem_time < age_threshold
            is_low_importance = memory.importance < self.config.importance_threshold
            is_low_access = memory.access_count <= self.config.access_threshold
            is_in_cluster = memory.memory_id in clustered_ids

            # Deprecate if: (low quality AND old) OR (in cluster AND low quality)
            if (is_low_importance and is_low_access and is_old) or (
                is_in_cluster and is_low_importance and is_low_access
            ):
                deprecated_ids.append(memory.memory_id)

        return deprecated_ids

    def _empty_result(self, started_at: datetime) -> ConsolidationResult:
        """Create an empty result when no consolidation occurs."""
        completed_at = datetime.now(timezone.utc)
        return ConsolidationResult(
            total_memories_processed=0,
            clusters_formed=0,
            semantic_memories_created=0,
            memories_deprecated=0,
            created_semantic_ids=[],
            deprecated_memory_ids=[],
            cluster_summaries=[],
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )
