"""
Memory introspection and evaluation metrics system.

Provides comprehensive evaluation framework for memory system quality:
1. Recall@k and Precision@k metrics
2. Drift rate tracking
3. Contradiction count monitoring
4. Synthetic dataset generation
5. Benchmark suite for continuous evaluation
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..core.models import MemoryItem, coerce_datetime


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    RECALL = "recall"
    PRECISION = "precision"
    F1_SCORE = "f1_score"
    MRR = "mrr"  # Mean Reciprocal Rank
    NDCG = "ndcg"  # Normalized Discounted Cumulative Gain
    DRIFT_RATE = "drift_rate"
    CONTRADICTION_COUNT = "contradiction_count"
    STALENESS_RATIO = "staleness_ratio"
    DIVERSITY = "diversity"


@dataclass
class EvaluationQuery:
    """A single evaluation query with ground truth."""

    query_id: str
    query_text: str
    relevant_memory_ids: list[str]  # Ground truth relevant memories
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of a single query evaluation."""

    query_id: str

    # Retrieved results
    retrieved_memory_ids: list[str]

    # Metrics
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    f1_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float | None = None
    ndcg: float | None = None

    # Diagnostics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregate results from benchmark evaluation."""

    # Aggregate metrics
    avg_recall_at_k: dict[int, float] = field(default_factory=dict)
    avg_precision_at_k: dict[int, float] = field(default_factory=dict)
    avg_f1_at_k: dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_ndcg: float = 0.0

    # System health metrics
    drift_rate: float = 0.0
    contradiction_count: int = 0
    staleness_ratio: float = 0.0
    diversity_score: float = 0.0

    # Per-query results
    query_results: list[EvaluationResult] = field(default_factory=list)

    # Metadata
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_queries: int = 0
    total_memories: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "avg_recall_at_k": self.avg_recall_at_k,
            "avg_precision_at_k": self.avg_precision_at_k,
            "avg_f1_at_k": self.avg_f1_at_k,
            "avg_mrr": self.avg_mrr,
            "avg_ndcg": self.avg_ndcg,
            "drift_rate": self.drift_rate,
            "contradiction_count": self.contradiction_count,
            "staleness_ratio": self.staleness_ratio,
            "diversity_score": self.diversity_score,
            "evaluated_at": str(self.evaluated_at),
            "total_queries": self.total_queries,
            "total_memories": self.total_memories,
        }


class MemoryEvaluator:
    """
    Comprehensive evaluation system for memory quality.

    Measures memory system performance using standard IR metrics
    and custom memory-specific health indicators.
    """

    def __init__(self, memory_core: Any | None = None):
        """Initialize evaluator."""
        self.memory_core = memory_core
        self.evaluation_history: list[BenchmarkResult] = []

    async def evaluate_query(
        self,
        query: EvaluationQuery,
        retrieved_memories: list[MemoryItem] | None = None,
        k_values: list[int] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a single query's retrieval results.

        Args:
            query: Evaluation query with ground truth
            retrieved_memories: Retrieved memory results
            k_values: Values of k for @k metrics

        Returns:
            EvaluationResult with metrics
        """
        if retrieved_memories is None:
            if self.memory_core is None:
                raise ValueError("memory_core is required when retrieved_memories is not provided")
            retrieved_memories = await self.memory_core.recall(query.query_text)

        return self.evaluate_query_with_retrieved(query, retrieved_memories, k_values)

    def evaluate_query_with_retrieved(
        self,
        query: EvaluationQuery,
        retrieved_memories: list[MemoryItem],
        k_values: list[int] | None = None,
    ) -> EvaluationResult:
        """Evaluate a query using already retrieved memories."""
        resolved_k_values = k_values or [1, 5, 10, 20]
        result = EvaluationResult(
            query_id=query.query_id,
            retrieved_memory_ids=[],
        )

        # Extract retrieved IDs
        retrieved_ids = [m.memory_id for m in retrieved_memories]
        result.retrieved_memory_ids = retrieved_ids

        # Ground truth
        relevant_ids = set(query.relevant_memory_ids)

        # Calculate metrics at different k values
        for k in resolved_k_values:
            recall = self._compute_recall_at_k(query.relevant_memory_ids, retrieved_ids, k)
            precision = self._compute_precision_at_k(query.relevant_memory_ids, retrieved_ids, k)
            result.recall_at_k[k] = recall
            result.precision_at_k[k] = precision

            # F1@k
            f1 = 2 * (precision * recall) / (precision + recall) if recall + precision > 0 else 0.0
            result.f1_at_k[k] = f1

        # Overall diagnostics
        retrieved_set = set(retrieved_ids)
        result.true_positives = len(retrieved_set & relevant_ids)
        result.false_positives = len(retrieved_set - relevant_ids)
        result.false_negatives = len(relevant_ids - retrieved_set)

        # MRR (Mean Reciprocal Rank)
        result.mrr = self._compute_mrr(query.relevant_memory_ids, retrieved_ids)

        # NDCG (Normalized Discounted Cumulative Gain)
        result.ndcg = self._calculate_ndcg(retrieved_ids, relevant_ids)

        return result

    def evaluate_benchmark(
        self,
        queries: list[EvaluationQuery],
        retrieval_function: Any,  # Function that takes query and returns memories
        all_memories: list[MemoryItem],
        k_values: list[int] | None = None,
    ) -> BenchmarkResult:
        """
        Run full benchmark evaluation.

        Args:
            queries: List of evaluation queries
            retrieval_function: Function to retrieve memories for a query
            all_memories: All memories in the system
            k_values: Values of k for @k metrics

        Returns:
            BenchmarkResult with aggregate metrics
        """
        benchmark = BenchmarkResult()
        benchmark.total_queries = len(queries)
        benchmark.total_memories = len(all_memories)
        resolved_k_values = k_values or [1, 5, 10, 20]

        # Evaluate each query
        query_results = []
        for query in queries:
            # Retrieve memories
            retrieved = retrieval_function(query.query_text)

            # Evaluate
            result = self.evaluate_query_with_retrieved(query, retrieved, resolved_k_values)
            query_results.append(result)

        benchmark.query_results = query_results

        # Calculate aggregate metrics
        for k in resolved_k_values:
            recalls = [r.recall_at_k.get(k, 0.0) for r in query_results]
            precisions = [r.precision_at_k.get(k, 0.0) for r in query_results]
            f1s = [r.f1_at_k.get(k, 0.0) for r in query_results]

            benchmark.avg_recall_at_k[k] = float(np.mean(recalls))
            benchmark.avg_precision_at_k[k] = float(np.mean(precisions))
            benchmark.avg_f1_at_k[k] = float(np.mean(f1s))

        # Aggregate MRR and NDCG
        mrrs = [r.mrr for r in query_results if r.mrr is not None]
        ndcgs = [r.ndcg for r in query_results if r.ndcg is not None]

        benchmark.avg_mrr = float(np.mean(mrrs)) if mrrs else 0.0
        benchmark.avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

        # Calculate system health metrics
        benchmark.drift_rate = self.calculate_drift_rate(all_memories)
        benchmark.staleness_ratio = self.calculate_staleness_ratio(all_memories)
        benchmark.diversity_score = self.calculate_diversity(all_memories)

        # Store in history
        self.evaluation_history.append(benchmark)

        return benchmark

    def _compute_recall_at_k(
        self, relevant_ids: list[str], retrieved_ids: list[str], k: int
    ) -> float:
        """Compute recall@k."""
        if k <= 0:
            return 0.0
        if not relevant_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        retrieved_at_k = retrieved_ids[:k]
        tp = sum(1 for mem_id in retrieved_at_k if mem_id in relevant_set)
        return tp / len(relevant_set)

    def _compute_precision_at_k(
        self, relevant_ids: list[str], retrieved_ids: list[str], k: int
    ) -> float:
        """Compute precision@k."""
        if k <= 0:
            return 0.0
        relevant_set = set(relevant_ids)
        retrieved_at_k = retrieved_ids[:k]
        tp = sum(1 for mem_id in retrieved_at_k if mem_id in relevant_set)
        return tp / k

    def _compute_mrr(self, relevant_ids: list[str], retrieved_ids: list[str]) -> float:
        """Compute Mean Reciprocal Rank."""
        relevant_set = set(relevant_ids)
        for rank, mem_id in enumerate(retrieved_ids, start=1):
            if mem_id in relevant_set:
                return 1.0 / rank
        return 0.0

    def _calculate_mrr(self, retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        return self._compute_mrr(list(relevant_ids), retrieved_ids)

    def _calculate_ndcg(
        self, retrieved_ids: list[str], relevant_ids: set[str], k: int | None = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if k is None:
            k = len(retrieved_ids)

        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, mem_id in enumerate(retrieved_ids[:k], start=1):
            rel = 1.0 if mem_id in relevant_ids else 0.0
            dcg += rel / np.log2(i + 1)

        # IDCG (Ideal DCG)
        ideal_ranks = min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_ranks + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_drift_rate(self, memories: list[MemoryItem], time_window_days: int = 30) -> float:
        """
        Calculate embedding drift rate.

        Measures how much memory embeddings change over time,
        indicating concept drift or inconsistency.

        Args:
            memories: Memories to analyze
            time_window_days: Time window for comparison

        Returns:
            Drift rate (0 = stable, 1 = high drift)
        """
        if len(memories) < 2:
            return 0.0

        # Group memories by time windows
        now = datetime.now(timezone.utc)

        recent_memories = []
        old_memories = []

        for memory in memories:
            if not memory.embedding:
                continue

            mem_time = coerce_datetime(memory.timestamp, default=None)
            if mem_time is None:
                continue
            days_old = (now - mem_time).days

            if days_old < time_window_days:
                recent_memories.append(memory.embedding)
            else:
                old_memories.append(memory.embedding)

        if not recent_memories or not old_memories:
            return 0.0

        # Calculate centroid shift
        recent_centroid = np.mean(recent_memories, axis=0)
        old_centroid = np.mean(old_memories, axis=0)

        # Euclidean distance between centroids (normalized)
        drift = float(np.linalg.norm(recent_centroid - old_centroid).item())

        # Normalize to [0, 1] range
        drift_rate = drift / 2.0
        if drift_rate > 1.0:
            drift_rate = 1.0

        return drift_rate

    def calculate_staleness_ratio(
        self, memories: list[MemoryItem], staleness_threshold_days: int = 90
    ) -> float:
        """
        Calculate ratio of stale memories.

        Args:
            memories: Memories to analyze
            staleness_threshold_days: Days after which memory is considered stale

        Returns:
            Ratio of stale memories (0-1)
        """
        if not memories:
            return 0.0

        now = datetime.now(timezone.utc)
        stale_count = 0

        for memory in memories:
            last_accessed = coerce_datetime(memory.last_accessed, default=None)
            if last_accessed is None:
                continue
            days_since_access = (now - last_accessed).days

            if days_since_access > staleness_threshold_days:
                stale_count += 1

        return stale_count / len(memories)

    def calculate_diversity(self, memories: list[MemoryItem]) -> float:
        """
        Calculate diversity of memory embeddings.

        Higher diversity = more varied information stored.

        Args:
            memories: Memories to analyze

        Returns:
            Diversity score (0-1)
        """
        embeddings = [m.embedding for m in memories if m.embedding]

        if len(embeddings) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(np.array(embeddings[i]) - np.array(embeddings[j]))
                distances.append(dist)

        if not distances:
            return 0.0

        # Average distance as diversity measure
        avg_distance = float(np.mean(distances).item())

        # Normalize to [0, 1] range
        diversity = avg_distance / 2.0
        if diversity > 1.0:
            diversity = 1.0

        return diversity

    def save_benchmark(self, benchmark: BenchmarkResult, filepath: Path):
        """Save benchmark results to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(benchmark.to_dict(), f, indent=2)

    def compare_benchmarks(
        self, baseline: BenchmarkResult, current: BenchmarkResult
    ) -> dict[str, Any]:
        """
        Compare two benchmark results.

        Args:
            baseline: Baseline benchmark
            current: Current benchmark

        Returns:
            Dictionary of metric changes
        """
        comparison = {}

        # Compare recall@k
        for k in baseline.avg_recall_at_k:
            if k in current.avg_recall_at_k:
                baseline_val = baseline.avg_recall_at_k[k]
                current_val = current.avg_recall_at_k[k]
                change = current_val - baseline_val
                comparison[f"recall@{k}_change"] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change": change,
                    "percent_change": (change / baseline_val * 100) if baseline_val > 0 else 0,
                }

        # Compare precision@k
        for k in baseline.avg_precision_at_k:
            if k in current.avg_precision_at_k:
                baseline_val = baseline.avg_precision_at_k[k]
                current_val = current.avg_precision_at_k[k]
                change = current_val - baseline_val
                comparison[f"precision@{k}_change"] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change": change,
                    "percent_change": (change / baseline_val * 100) if baseline_val > 0 else 0,
                }

        # Compare system metrics
        for metric in ["drift_rate", "staleness_ratio", "diversity_score"]:
            baseline_val = getattr(baseline, metric, 0.0)
            current_val = getattr(current, metric, 0.0)
            change = current_val - baseline_val
            comparison[f"{metric}_change"] = {
                "baseline": baseline_val,
                "current": current_val,
                "change": change,
            }

        return comparison


class SyntheticDatasetGenerator:
    """
    Generate synthetic evaluation datasets for memory testing.

    Creates realistic queries and ground truth for benchmarking.
    """

    def __init__(self, seed: int = 42, llm: Any | None = None):
        """Initialize generator with random seed."""
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.llm = llm

    def _sample_memories(self, memories: list[MemoryItem], num_queries: int) -> list[MemoryItem]:
        sampled_indices = self.rng.choice(
            len(memories), size=min(num_queries, len(memories)), replace=False
        )
        return [memories[idx] for idx in sampled_indices]

    @staticmethod
    def _build_query_text(memory: MemoryItem) -> str:
        sentences = memory.content.split(".")
        return sentences[0] + "." if sentences else memory.content[:50]

    def _collect_relevant_ids(
        self,
        base_memory: MemoryItem,
        memories: list[MemoryItem],
        relevance_threshold: float,
    ) -> list[str]:
        relevant_ids = [base_memory.memory_id]
        if not base_memory.embedding:
            return relevant_ids

        for other_memory in memories:
            if other_memory.memory_id == base_memory.memory_id:
                continue
            if not other_memory.embedding:
                continue

            similarity = self._cosine_similarity(base_memory.embedding, other_memory.embedding)
            if similarity >= relevance_threshold:
                relevant_ids.append(other_memory.memory_id)

        return relevant_ids

    def generate_queries(
        self, memories: list[MemoryItem], num_queries: int = 50, relevance_threshold: float = 0.7
    ) -> list[EvaluationQuery]:
        """
        Generate evaluation queries from existing memories.

        Args:
            memories: Existing memories to base queries on
            num_queries: Number of queries to generate
            relevance_threshold: Similarity threshold for relevance

        Returns:
            List of evaluation queries with ground truth
        """
        if len(memories) < 10:
            raise ValueError("Need at least 10 memories to generate queries")

        queries: list[EvaluationQuery] = []

        for base_memory in self._sample_memories(memories, num_queries):
            query_text = self._build_query_text(base_memory)
            relevant_ids = self._collect_relevant_ids(base_memory, memories, relevance_threshold)
            queries.append(
                EvaluationQuery(
                    query_id=f"query_{len(queries) + 1}",
                    query_text=query_text,
                    relevant_memory_ids=relevant_ids,
                    metadata={"base_memory_id": base_memory.memory_id},
                )
            )

        return queries

    async def generate_queries_from_memories(
        self, memories: list[MemoryItem], queries_per_memory: int = 2
    ) -> list[EvaluationQuery]:
        """
        Generate synthetic queries for each memory, optionally using an LLM.

        Args:
            memories: Memories to create queries for
            queries_per_memory: Number of queries to generate per memory

        Returns:
            List of evaluation queries
        """
        if not memories:
            return []

        queries: list[EvaluationQuery] = []

        for memory in memories:
            generated_queries: list[str] = []
            if self.llm is not None:
                prompt = (
                    "Generate short questions that could be answered by the memory below.\n"
                    f"Memory: {memory.content}\n"
                    f"Return {queries_per_memory} questions, one per line."
                )
                response = await self.llm.ainvoke(prompt)
                content = getattr(response, "content", str(response))
                generated_queries = [line.strip() for line in content.splitlines() if line.strip()]

            if not generated_queries:
                # Fallback: simple template-based query
                generated_queries = [f"What is mentioned about: {memory.content[:40]}?"]

            for query_text in generated_queries[:queries_per_memory]:
                queries.append(
                    EvaluationQuery(
                        query_id=f"synthetic_{len(queries) + 1}",
                        query_text=query_text,
                        relevant_memory_ids=[memory.memory_id],
                        metadata={"source_memory_id": memory.memory_id},
                    )
                )

        return queries

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def save_queries(self, queries: list[EvaluationQuery], filepath: Path):
        """Save queries to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        queries_data = [
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "relevant_memory_ids": q.relevant_memory_ids,
                "metadata": q.metadata,
            }
            for q in queries
        ]

        with open(filepath, "w") as f:
            json.dump(queries_data, f, indent=2)

    def load_queries(self, filepath: Path) -> list[EvaluationQuery]:
        """Load queries from file."""
        with open(filepath) as f:
            queries_data = json.load(f)

        return [
            EvaluationQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                relevant_memory_ids=q["relevant_memory_ids"],
                metadata=q.get("metadata", {}),
            )
            for q in queries_data
        ]
