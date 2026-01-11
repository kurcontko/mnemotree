from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from networkx import Graph, connected_components
from sklearn.cluster import DBSCAN, KMeans

from ..core.models import MemoryItem
from .summarizer import Summarizer


@dataclass
class ClusteringResult:
    """Results from memory clustering."""

    cluster_ids: list[int]  # Cluster assignment for each memory
    centroids: list[np.ndarray] | None  # Cluster centers (for vector-based)
    cluster_sizes: dict[int, int]  # Number of memories in each cluster
    cluster_summaries: dict[int, str]  # Summary of each cluster's content


class MemoryClusterer:
    """Handles clustering of memories using vector or graph-based approaches."""

    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer

    async def cluster_memories(
        self,
        memories: list[MemoryItem],
        method: Literal["vector_dbscan", "vector_kmeans", "graph_community"] = "vector_dbscan",
        *,
        eps: float = 0.3,  # For DBSCAN
        min_samples: int = 5,  # For DBSCAN
        n_clusters: int | None = None,  # For K-means
        similarity_threshold: float = 0.7,  # For graph-based
    ) -> ClusteringResult:
        """
        Cluster memories using specified method.

        Args:
            memories: List of memories to cluster
            method: Clustering method to use
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            n_clusters: Number of clusters for K-means
            similarity_threshold: Threshold for graph edge creation

        Returns:
            ClusteringResult containing cluster assignments and metadata
        """
        if not memories:
            return ClusteringResult([], [], {}, {})

        if method.startswith("vector_"):
            # Extract embeddings
            embeddings = np.array([m.embedding for m in memories])

            if method == "vector_dbscan":
                clusters = await self._dbscan_clustering(
                    embeddings, eps=eps, min_samples=min_samples
                )
            else:  # vector_kmeans
                if n_clusters is None:
                    n_clusters = max(2, len(memories) // 10)  # Heuristic
                clusters = await self._kmeans_clustering(embeddings, n_clusters=n_clusters)

        else:  # graph_community
            clusters = await self._graph_clustering(
                memories, similarity_threshold=similarity_threshold
            )

        # Get cluster metadata
        unique_clusters = set(clusters.cluster_ids)
        cluster_sizes = {c: clusters.cluster_ids.count(c) for c in unique_clusters}

        # Generate summaries for each cluster
        cluster_summaries = {}
        for cluster_id in unique_clusters:
            cluster_memories = [
                m for i, m in enumerate(memories) if clusters.cluster_ids[i] == cluster_id
            ]
            memory_texts = [f"- {m.content}" for m in cluster_memories]
            summary = await self.summarizer.summarize("\n".join(memory_texts))
            cluster_summaries[cluster_id] = summary

        return ClusteringResult(
            cluster_ids=clusters.cluster_ids,
            centroids=clusters.centroids,
            cluster_sizes=cluster_sizes,
            cluster_summaries=cluster_summaries,
        )

    async def _dbscan_clustering(
        self, embeddings: np.ndarray, eps: float, min_samples: int
    ) -> ClusteringResult:
        """Perform DBSCAN clustering on memory embeddings."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_ids = dbscan.fit_predict(embeddings)

        # Calculate cluster centroids
        centroids = []
        for cluster_id in set(cluster_ids):
            if cluster_id == -1:  # Noise points
                continue
            mask = cluster_ids == cluster_id
            centroid = embeddings[mask].mean(axis=0)
            centroids.append(centroid)

        return ClusteringResult(
            cluster_ids=cluster_ids.tolist(),
            centroids=centroids,
            cluster_sizes={},  # Will be filled later
            cluster_summaries={},  # Will be filled later
        )

    async def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Perform K-means clustering on memory embeddings."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(embeddings)

        return ClusteringResult(
            cluster_ids=cluster_ids.tolist(),
            centroids=kmeans.cluster_centers_.tolist(),
            cluster_sizes={},  # Will be filled later
            cluster_summaries={},  # Will be filled later
        )

    async def _graph_clustering(
        self, memories: list[MemoryItem], similarity_threshold: float
    ) -> ClusteringResult:
        """Perform graph-based clustering using community detection."""
        # Create similarity graph
        graph = Graph()
        for i, mem1 in enumerate(memories):
            graph.add_node(i)
            for j, mem2 in enumerate(memories[i + 1 :], i + 1):
                similarity = np.dot(mem1.embedding, mem2.embedding)
                if similarity >= similarity_threshold:
                    graph.add_edge(i, j)

        # Find connected components as clusters
        clusters = list(connected_components(graph))

        # Convert to cluster IDs format
        cluster_ids = [-1] * len(memories)  # Default to noise
        for cluster_idx, cluster in enumerate(clusters):
            for node_idx in cluster:
                cluster_ids[node_idx] = cluster_idx

        return ClusteringResult(
            cluster_ids=cluster_ids,
            centroids=None,  # No centroids for graph-based clustering
            cluster_sizes={},  # Will be filled later
            cluster_summaries={},  # Will be filled later
        )
