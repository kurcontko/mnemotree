"""
Adaptive importance and decay system with spaced repetition.

Implements:
1. Dynamic decay based on usage patterns, recency, and novelty
2. Spaced repetition scheduling for high-value memories
3. Novelty detection and importance boosting
4. Context-aware importance adjustment
5. Personalized decay curves per memory type
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ..core.models import MemoryItem, MemoryType, coerce_datetime


class DecayProfile(str, Enum):
    """Decay behavior profiles."""

    STABLE = "stable"  # Slow decay (facts, knowledge)
    MODERATE = "moderate"  # Normal decay (general memories)
    RAPID = "rapid"  # Fast decay (transient info)
    ADAPTIVE = "adaptive"  # Context-dependent decay


class NoveltyLevel(str, Enum):
    """Novelty assessment levels."""

    NEW = "new"  # Never seen before
    FAMILIAR = "familiar"  # Seen a few times
    ROUTINE = "routine"  # Frequently encountered
    REDUNDANT = "redundant"  # Overly common


@dataclass
class SpacedRepetitionSchedule:
    """
    Spaced repetition schedule for memory reinforcement.

    Based on SuperMemo SM-2 algorithm with modifications.
    """

    memory_id: str
    easiness_factor: float = 2.5  # How easy is this memory to recall (1.3-2.5)
    interval_days: float = 1.0  # Days until next review
    repetition_count: int = 0  # Number of successful recalls
    last_review: datetime = None
    next_review: datetime = None

    def schedule_next_review(self, quality: int):
        """
        Schedule next review based on recall quality.

        Args:
            quality: Quality of recall (0-5)
                0-2: Failure (reset)
                3: Difficult
                4: Good
                5: Perfect
        """
        now = datetime.now(timezone.utc)
        self.last_review = now

        # Update easiness factor
        self.easiness_factor = max(
            1.3, self.easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        )

        if quality < 3:
            # Failed recall - reset
            self.repetition_count = 0
            self.interval_days = 1.0
        else:
            # Successful recall - increase interval
            self.repetition_count += 1

            if self.repetition_count == 1:
                self.interval_days = 1.0
            elif self.repetition_count == 2:
                self.interval_days = 6.0
            else:
                self.interval_days = self.interval_days * self.easiness_factor

        self.next_review = now + timedelta(days=self.interval_days)


@dataclass
class DecayParameters:
    """Parameters for adaptive decay calculation."""

    # Base decay rates per memory type
    base_decay_rates: dict[MemoryType, float]

    # Usage pattern weights
    access_frequency_weight: float = 0.3
    recency_weight: float = 0.4
    novelty_weight: float = 0.3

    # Decay modifiers
    high_importance_threshold: float = 0.7  # Slow decay above this
    low_access_threshold: int = 2  # Fast decay below this
    staleness_days: int = 30  # Accelerate decay after this

    @classmethod
    def default(cls) -> "DecayParameters":
        """Create default decay parameters."""
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


class AdaptiveImportanceSystem:
    """
    Adaptive importance and decay system.

    Dynamically adjusts memory importance and decay based on:
    - Usage patterns (access frequency, recency)
    - Novelty (how unique is this information)
    - Context (relationship to other memories)
    - Memory type (different types decay differently)
    - Spaced repetition (reinforce valuable memories)
    """

    def __init__(
        self,
        decay_params: DecayParameters | None = None,
        enable_spaced_repetition: bool = True,
        novelty_window_days: int = 7,
    ):
        """
        Initialize adaptive importance system.

        Args:
            decay_params: Decay parameters
            enable_spaced_repetition: Enable spaced repetition for high-value memories
            novelty_window_days: Days to track for novelty detection
        """
        self.decay_params = decay_params or DecayParameters.default()
        self.enable_spaced_repetition = enable_spaced_repetition
        self.novelty_window_days = novelty_window_days

        # Spaced repetition tracking
        self.sr_schedules: dict[str, SpacedRepetitionSchedule] = {}

        # Novelty tracking
        self._content_hashes: dict[
            str, list[tuple[str, datetime]]
        ] = {}  # hash -> [(memory_id, timestamp)]

    def calculate_adaptive_decay(
        self, memory: MemoryItem, current_time: datetime | None = None
    ) -> float:
        """
        Calculate adaptive decay rate for a memory.

        Returns:
            Decay rate (higher = faster decay)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Get base decay rate for memory type
        base_decay = self.decay_params.base_decay_rates.get(memory.memory_type, 0.01)

        # Calculate usage pattern modifier
        usage_modifier = self._calculate_usage_modifier(memory, current_time)

        # Calculate recency modifier
        recency_modifier = self._calculate_recency_modifier(memory, current_time)

        # Calculate novelty modifier
        novelty_modifier = self._calculate_novelty_modifier(memory)

        # Combine modifiers
        combined_modifier = (
            self.decay_params.access_frequency_weight * usage_modifier
            + self.decay_params.recency_weight * recency_modifier
            + self.decay_params.novelty_weight * novelty_modifier
        )

        # Apply importance scaling
        importance_factor = 1.0
        if memory.importance >= self.decay_params.high_importance_threshold:
            # High importance = slower decay
            importance_factor = 0.5
        elif memory.importance < 0.3:
            # Low importance = faster decay
            importance_factor = 2.0

        # Final decay rate
        adaptive_decay = base_decay * combined_modifier * importance_factor

        return max(0.0, min(1.0, adaptive_decay))

    def _calculate_usage_modifier(self, memory: MemoryItem, current_time: datetime) -> float:
        """
        Calculate modifier based on usage patterns.

        High access = lower modifier (slower decay)
        Low access = higher modifier (faster decay)
        """
        access_count = memory.access_count

        if access_count == 0:
            return 2.0  # Never accessed = faster decay
        elif access_count < self.decay_params.low_access_threshold:
            return 1.5  # Rarely accessed
        elif access_count < 10:
            return 1.0  # Normal access
        elif access_count < 50:
            return 0.7  # Frequently accessed
        else:
            return 0.5  # Very frequently accessed

    def _calculate_recency_modifier(self, memory: MemoryItem, current_time: datetime) -> float:
        """
        Calculate modifier based on recency.

        Recent access = lower modifier (slower decay)
        Old access = higher modifier (faster decay)
        """
        last_accessed = coerce_datetime(
            memory.last_accessed,
            default=datetime.now(timezone.utc),
        )

        days_since_access = (current_time - last_accessed).days

        if days_since_access < 1:
            return 0.5  # Very recent
        elif days_since_access < 7:
            return 0.8  # Recent
        elif days_since_access < 30:
            return 1.0  # Moderate
        elif days_since_access < self.decay_params.staleness_days:
            return 1.3  # Getting old
        else:
            return 2.0  # Stale

    def _calculate_novelty_modifier(self, memory: MemoryItem) -> float:
        """
        Calculate modifier based on novelty.

        Novel information = lower modifier (slower decay)
        Redundant information = higher modifier (faster decay)
        """
        novelty_level = self.assess_novelty(memory)

        if novelty_level == NoveltyLevel.NEW:
            return 0.6  # Novel = preserve
        elif novelty_level == NoveltyLevel.FAMILIAR:
            return 0.9  # Somewhat novel
        elif novelty_level == NoveltyLevel.ROUTINE:
            return 1.1  # Common
        else:  # REDUNDANT
            return 1.5  # Very common = faster decay

    def assess_novelty(self, memory: MemoryItem) -> NoveltyLevel:
        """
        Assess novelty of a memory based on content similarity to recent memories.

        Args:
            memory: Memory to assess

        Returns:
            NoveltyLevel assessment
        """
        # Simple hash-based novelty check
        content_hash = hash(memory.content.lower()[:200])
        hash_key = str(content_hash)

        if hash_key not in self._content_hashes:
            self._content_hashes[hash_key] = []

        # Clean old entries
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.novelty_window_days)
        self._content_hashes[hash_key] = [
            (mid, ts) for mid, ts in self._content_hashes[hash_key] if ts > cutoff
        ]

        # Count recent similar content
        similar_count = len(self._content_hashes[hash_key])

        # Add current memory
        self._content_hashes[hash_key].append((memory.memory_id, datetime.now(timezone.utc)))

        # Assess novelty
        if similar_count == 0:
            return NoveltyLevel.NEW
        elif similar_count < 3:
            return NoveltyLevel.FAMILIAR
        elif similar_count < 10:
            return NoveltyLevel.ROUTINE
        else:
            return NoveltyLevel.REDUNDANT

    def update_importance(self, memory: MemoryItem, current_time: datetime | None = None) -> float:
        """
        Update memory importance based on adaptive decay.

        Args:
            memory: Memory to update
            current_time: Current timestamp

        Returns:
            New importance value
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Calculate adaptive decay
        decay_rate = self.calculate_adaptive_decay(memory, current_time)

        # Apply decay
        last_accessed = coerce_datetime(memory.last_accessed, default=current_time)

        time_diff = (current_time - last_accessed).total_seconds()
        decay_amount = decay_rate * (time_diff / 86400.0)  # Convert to days

        new_importance = max(0.0, memory.importance - decay_amount)

        # Check for spaced repetition boost
        if self.enable_spaced_repetition and memory.memory_id in self.sr_schedules:
            schedule = self.sr_schedules[memory.memory_id]
            if schedule.next_review and current_time >= schedule.next_review:
                # Due for review - boost importance
                new_importance = min(1.0, new_importance + 0.1)

        return new_importance

    def register_for_spaced_repetition(self, memory: MemoryItem, initial_quality: int = 4):
        """
        Register a high-value memory for spaced repetition.

        Args:
            memory: Memory to register
            initial_quality: Initial recall quality (0-5)
        """
        if memory.memory_id in self.sr_schedules:
            return  # Already registered

        schedule = SpacedRepetitionSchedule(
            memory_id=memory.memory_id, last_review=datetime.now(timezone.utc)
        )
        schedule.schedule_next_review(initial_quality)

        self.sr_schedules[memory.memory_id] = schedule

    def record_recall(self, memory_id: str, quality: int):
        """
        Record a recall event and update spaced repetition schedule.

        Args:
            memory_id: ID of recalled memory
            quality: Quality of recall (0-5)
        """
        if memory_id not in self.sr_schedules:
            return

        schedule = self.sr_schedules[memory_id]
        schedule.schedule_next_review(quality)

    def get_due_reviews(self, current_time: datetime | None = None) -> list[str]:
        """
        Get memories due for spaced repetition review.

        Args:
            current_time: Current timestamp

        Returns:
            List of memory IDs due for review
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        due_ids = []
        for memory_id, schedule in self.sr_schedules.items():
            if schedule.next_review and current_time >= schedule.next_review:
                due_ids.append(memory_id)

        return due_ids

    def boost_importance_for_novelty(self, memory: MemoryItem, boost_amount: float = 0.1) -> float:
        """
        Boost importance for novel memories.

        Args:
            memory: Memory to evaluate
            boost_amount: Amount to boost if novel

        Returns:
            New importance value
        """
        novelty = self.assess_novelty(memory)

        if novelty == NoveltyLevel.NEW:
            return min(1.0, memory.importance + boost_amount)
        elif novelty == NoveltyLevel.FAMILIAR:
            return min(1.0, memory.importance + boost_amount * 0.5)
        else:
            return memory.importance

    def get_decay_statistics(self) -> dict[str, Any]:
        """Get statistics about the adaptive system."""
        return {
            "spaced_repetition_enabled": self.enable_spaced_repetition,
            "memories_in_sr": len(self.sr_schedules),
            "due_reviews": len(self.get_due_reviews()),
            "novelty_window_days": self.novelty_window_days,
            "tracked_content_hashes": len(self._content_hashes),
        }
