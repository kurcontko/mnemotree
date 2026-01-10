import math
from datetime import datetime, timezone
from typing import Literal, overload

import numpy as np

from .models import MemoryItem, coerce_datetime


def cosine_similarity(
    vec1: list[float] | None,
    vec2: list[float] | None,
) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)

    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm1 * norm2))


class MemoryScoring:
    """Simplified memory scoring based on recency, relevance, and importance."""

    def __init__(
        self,
        importance_weight: float = 0.4,
        recency_weight: float = 0.4,
        relevance_weight: float = 0.2,
        score_threshold: float = 0.0,
        recency_stability_seconds: float = 24 * 3600,
        recency_power: float = -0.5,
    ):
        self.weights = {
            "importance": importance_weight,
            "recency": recency_weight,
            "relevance": relevance_weight,
        }
        self._normalize_weights()
        self.score_threshold = score_threshold
        self.recency_stability_seconds = recency_stability_seconds
        self.recency_power = recency_power

    @overload
    def calculate_memory_score(
        self,
        memory: MemoryItem,
        current_time: datetime | None = ...,
        query_embedding: list[float] | None = ...,
        return_components: Literal[False] = ...,
    ) -> float: ...

    @overload
    def calculate_memory_score(
        self,
        memory: MemoryItem,
        current_time: datetime | None = ...,
        query_embedding: list[float] | None = ...,
        return_components: Literal[True] = ...,
    ) -> tuple[float, dict[str, float]]: ...

    def calculate_memory_score(
        self,
        memory: MemoryItem,
        current_time: datetime | None = None,
        query_embedding: list[float] | None = None,
        return_components: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        """
        Score formula:
        score = w_recency * recency + w_relevance * relevance + w_importance * importance

        recency = ((time_diff / stability) + 1) ** power
        relevance = max(0, cosine_similarity)
        importance = clamp(importance + log(access_count + 1) * 0.05)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        memory_time = coerce_datetime(memory.timestamp, default=current_time)

        components = {
            "recency": self._calculate_recency_score(memory_time, current_time),
            "relevance": self._calculate_relevance_score(memory, query_embedding),
            "importance": self._calculate_importance_score(memory),
        }

        score = sum(self.weights[name] * value for name, value in components.items())

        if return_components:
            return score, components
        return score

    def _calculate_importance_score(self, memory: MemoryItem) -> float:
        """Base importance with a log-scaled access boost."""
        access_boost = min(0.2, math.log(memory.access_count + 1) * 0.05)
        return min(1.0, max(0.0, memory.importance + access_boost))

    def _calculate_recency_score(
        self,
        memory_time: datetime,
        current_time: datetime,
    ) -> float:
        """Power-law recency decay with stability normalization."""
        time_diff = max(0.0, (current_time - memory_time).total_seconds())
        if self.recency_stability_seconds <= 0:
            return 0.0
        normalized = time_diff / self.recency_stability_seconds
        recency = (normalized + 1.0) ** self.recency_power
        return min(1.0, max(0.0, recency))

    def _calculate_relevance_score(
        self,
        memory: MemoryItem,
        query_embedding: list[float] | None,
    ) -> float:
        """Positive cosine similarity in [0, 1]."""
        if not query_embedding or not memory.embedding:
            return 0.0
        similarity = cosine_similarity(memory.embedding, query_embedding)
        return max(0.0, min(1.0, similarity))

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def filter_memories_by_score(
        self,
        memories: list[MemoryItem],
        query_embedding: list[float] | None = None,
    ) -> list[MemoryItem]:
        """Filter memories based on score threshold."""
        current_time = datetime.now(timezone.utc)
        filtered_memories = []
        for memory in memories:
            score = self.calculate_memory_score(
                memory,
                current_time,
                query_embedding,
            )
            if score >= self.score_threshold:
                filtered_memories.append(memory)
        return filtered_memories
