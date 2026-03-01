"""
Adaptive decay system with FSRS-based spaced repetition.

Moved from experimental/adaptive_decay.py into core so the scoring pipeline
can call ``adjust_decay_config`` without reaching into an experimental package.

Key integration point:
    ``AdaptiveDecaySystem.adjust_decay_config(memory, base_config, current_time)``
    is called by ``MemoryScoring._calculate_importance_score()`` when wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .decay import DecayConfig, compute_decayed_importance
from .models import MemoryItem, MemoryType, coerce_datetime
from .scoring import cosine_similarity

# ---------------------------------------------------------------------------
# Enums (kept as-is from experimental)
# ---------------------------------------------------------------------------


class DecayProfile(str):
    """Decay behavior profiles."""

    STABLE = "stable"
    MODERATE = "moderate"
    RAPID = "rapid"
    ADAPTIVE = "adaptive"


class NoveltyLevel(str):
    """Novelty assessment levels."""

    NEW = "new"
    FAMILIAR = "familiar"
    ROUTINE = "routine"
    REDUNDANT = "redundant"


# ---------------------------------------------------------------------------
# FSRS-based spaced repetition (replaces SM-2)
# ---------------------------------------------------------------------------


@dataclass
class FSRSSchedule:
    """FSRS-style spaced repetition schedule.

    DSR model: difficulty (1-10), stability_seconds, last_review, repetition_count.

    Grade scale:
        1 = Again (reset)
        2 = Hard
        3 = Good
        4 = Easy
    """

    memory_id: str
    difficulty: float = 5.0  # 1-10 scale
    stability_seconds: float = 86400.0  # 1 day initial
    last_review: datetime | None = None
    repetition_count: int = 0
    next_review: datetime | None = None

    def schedule_next_review(self, grade: int, current_time: datetime | None = None):
        """Schedule next review based on recall grade (1-4)."""
        now = current_time or datetime.now(timezone.utc)
        self.last_review = now

        if grade == 1:
            # Again — reset
            self.repetition_count = 0
            self.stability_seconds = 60.0  # 1 minute
            self.difficulty = min(10.0, self.difficulty + 2.0)
        else:
            self.repetition_count += 1
            # Update difficulty
            delta = {2: 1.0, 3: 0.0, 4: -0.5}.get(grade, 0.0)
            self.difficulty = max(1.0, min(10.0, self.difficulty + delta))

            # Grow stability
            multiplier = {2: 1.2, 3: 2.5, 4: 3.5}.get(grade, 2.5)
            difficulty_factor = 1.0 / (1.0 + 0.1 * (self.difficulty - 5.0))
            self.stability_seconds = self.stability_seconds * multiplier * difficulty_factor

        # Cap at 365 days
        self.stability_seconds = min(self.stability_seconds, 365 * 86400)

        self.next_review = now + timedelta(seconds=self.stability_seconds)


# ---------------------------------------------------------------------------
# Decay parameters (kept from experimental)
# ---------------------------------------------------------------------------


@dataclass
class DecayParameters:
    """Parameters for adaptive decay calculation."""

    base_decay_rates: dict[MemoryType, float]

    access_frequency_weight: float = 0.3
    recency_weight: float = 0.4
    novelty_weight: float = 0.3

    high_importance_threshold: float = 0.7
    low_access_threshold: int = 2
    staleness_days: int = 30

    @classmethod
    def default(cls) -> DecayParameters:
        return cls(
            base_decay_rates={
                MemoryType.EPISODIC: 0.02,
                MemoryType.SEMANTIC: 0.005,
                MemoryType.PROCEDURAL: 0.001,
                MemoryType.WORKING: 0.1,
                MemoryType.AUTOBIOGRAPHICAL: 0.01,
                MemoryType.PROSPECTIVE: 0.03,
            }
        )


# ---------------------------------------------------------------------------
# Adaptive decay system
# ---------------------------------------------------------------------------


class AdaptiveDecaySystem:
    """Adaptive decay system that hooks into the scoring pipeline.

    Key method: ``adjust_decay_config(memory, base_config, current_time)``
    returns a modified ``DecayConfig`` with stability multiplied by adaptive factors.
    """

    def __init__(
        self,
        decay_params: DecayParameters | None = None,
        enable_spaced_repetition: bool = True,
        novelty_window_days: int = 7,
        novelty_similarity_threshold: float = 0.85,
    ):
        self.decay_params = decay_params or DecayParameters.default()
        self.enable_spaced_repetition = enable_spaced_repetition
        self.novelty_window_days = novelty_window_days
        self.novelty_similarity_threshold = novelty_similarity_threshold

        self.sr_schedules: dict[str, FSRSSchedule] = {}
        self._recent_embeddings: list[tuple[str, datetime, list[float]]] = []

    # ---- Scoring pipeline hook ----

    def adjust_decay_config(
        self,
        memory: MemoryItem,
        base_config: DecayConfig,
        current_time: datetime,
    ) -> DecayConfig:
        """Adjust stability_seconds via multipliers from usage, recency, novelty, importance."""
        usage_mult = self._usage_multiplier(memory)
        recency_mult = self._recency_multiplier(memory, current_time)
        novelty_mult = self._novelty_multiplier(memory)
        importance_mult = self._importance_multiplier(memory)

        combined = (
            self.decay_params.access_frequency_weight * usage_mult
            + self.decay_params.recency_weight * recency_mult
            + self.decay_params.novelty_weight * novelty_mult
        )
        combined *= importance_mult

        # Multiplier > 1 means slower decay (more stable), < 1 means faster
        new_stability = base_config.stability_seconds * combined
        new_stability = max(60.0, new_stability)  # At least 1 minute

        return DecayConfig(
            stability_seconds=new_stability,
            decay_power=base_config.decay_power,
            floor=base_config.floor,
            target_retention=base_config.target_retention,
        )

    # ---- Adaptive modifiers ----

    def _usage_multiplier(self, memory: MemoryItem) -> float:
        ac = memory.access_count
        if ac == 0:
            return 0.5
        elif ac < self.decay_params.low_access_threshold:
            return 0.7
        elif ac < 10:
            return 1.0
        elif ac < 50:
            return 1.4
        else:
            return 2.0

    def _recency_multiplier(self, memory: MemoryItem, current_time: datetime) -> float:
        last_accessed = coerce_datetime(memory.last_accessed, default=current_time)
        days = max(0.0, (current_time - last_accessed).total_seconds() / 86400.0)
        if days < 1:
            return 2.0
        elif days < 7:
            return 1.3
        elif days < 30:
            return 1.0
        elif days < self.decay_params.staleness_days:
            return 0.8
        else:
            return 0.5

    def _novelty_multiplier(self, memory: MemoryItem) -> float:
        level = self.assess_novelty(memory)
        if level == NoveltyLevel.NEW:
            return 1.7
        elif level == NoveltyLevel.FAMILIAR:
            return 1.1
        elif level == NoveltyLevel.ROUTINE:
            return 0.9
        else:
            return 0.7

    def _importance_multiplier(self, memory: MemoryItem) -> float:
        if memory.importance >= self.decay_params.high_importance_threshold:
            return 2.0  # high importance = much slower decay
        elif memory.importance < 0.3:
            return 0.5  # low importance = faster decay
        return 1.0

    # ---- Novelty assessment (embeddings-based) ----

    def assess_novelty(self, memory: MemoryItem) -> str:
        """Assess novelty using cosine similarity against recent embeddings."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.novelty_window_days)
        self._recent_embeddings = [
            (mid, ts, emb) for mid, ts, emb in self._recent_embeddings if ts > cutoff
        ]

        if memory.embedding is not None and memory.embedding:
            similar_count = sum(
                1
                for _, _, emb in self._recent_embeddings
                if cosine_similarity(memory.embedding, emb) >= self.novelty_similarity_threshold
            )
            self._recent_embeddings.append((memory.memory_id, now, list(memory.embedding)))
        else:
            # Fallback: hash-based
            similar_count = 0

        if similar_count == 0:
            return NoveltyLevel.NEW
        elif similar_count < 3:
            return NoveltyLevel.FAMILIAR
        elif similar_count < 10:
            return NoveltyLevel.ROUTINE
        else:
            return NoveltyLevel.REDUNDANT

    # ---- Importance update using new decay engine ----

    def update_importance(self, memory: MemoryItem, current_time: datetime | None = None) -> float:
        """Update memory importance using the power-law decay engine."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        from .decay import MEMORY_TYPE_DEFAULTS

        base_config = MEMORY_TYPE_DEFAULTS.get(memory.memory_type, DecayConfig())
        config = self.adjust_decay_config(memory, base_config, current_time)

        last_accessed = coerce_datetime(memory.last_accessed, default=current_time)
        elapsed = max(0.0, (current_time - last_accessed).total_seconds())

        new_importance = compute_decayed_importance(memory.importance, elapsed, config)

        # Check for spaced repetition boost
        if self.enable_spaced_repetition and memory.memory_id in self.sr_schedules:
            schedule = self.sr_schedules[memory.memory_id]
            if schedule.next_review and current_time >= schedule.next_review:
                new_importance = min(1.0, new_importance + 0.1)

        return new_importance

    # ---- Spaced repetition management ----

    def register_for_spaced_repetition(self, memory: MemoryItem, initial_grade: int = 3):
        if memory.memory_id in self.sr_schedules:
            return
        schedule = FSRSSchedule(memory_id=memory.memory_id)
        schedule.schedule_next_review(initial_grade)
        self.sr_schedules[memory.memory_id] = schedule

    def record_recall(self, memory_id: str, grade: int):
        if memory_id not in self.sr_schedules:
            return
        self.sr_schedules[memory_id].schedule_next_review(grade)

    def get_due_reviews(self, current_time: datetime | None = None) -> list[str]:
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        return [
            mid
            for mid, schedule in self.sr_schedules.items()
            if schedule.next_review and current_time >= schedule.next_review
        ]

    def boost_importance_for_novelty(self, memory: MemoryItem, boost_amount: float = 0.1) -> float:
        novelty = self.assess_novelty(memory)
        if novelty == NoveltyLevel.NEW:
            return min(1.0, memory.importance + boost_amount)
        elif novelty == NoveltyLevel.FAMILIAR:
            return min(1.0, memory.importance + boost_amount * 0.5)
        return memory.importance

    def get_decay_statistics(self) -> dict[str, Any]:
        return {
            "spaced_repetition_enabled": self.enable_spaced_repetition,
            "memories_in_sr": len(self.sr_schedules),
            "due_reviews": len(self.get_due_reviews()),
            "novelty_window_days": self.novelty_window_days,
            "tracked_embeddings": len(self._recent_embeddings),
        }
