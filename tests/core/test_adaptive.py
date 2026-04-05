"""Tests for mnemotree.core.adaptive — adaptive decay system and FSRS scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemotree.core.adaptive import (
    AdaptiveDecaySystem,
    FSRSSchedule,
    NoveltyLevel,
)
from mnemotree.core.decay import DecayConfig
from mnemotree.core.models import MemoryItem, MemoryType


def _memory(
    *,
    importance: float = 0.5,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    access_count: int = 0,
    last_accessed: datetime | None = None,
    embedding: list[float] | None = None,
) -> MemoryItem:
    now = datetime.now(timezone.utc)
    return MemoryItem(
        content="Test content",
        memory_type=memory_type,
        importance=importance,
        access_count=access_count,
        last_accessed=last_accessed or now,
        embedding=embedding,
    )


class TestAdaptiveDecaySystem:
    def test_adjust_decay_config_returns_decay_config(self):
        system = AdaptiveDecaySystem()
        memory = _memory()
        base = DecayConfig()
        now = datetime.now(timezone.utc)
        result = system.adjust_decay_config(memory, base, now)
        assert isinstance(result, DecayConfig)

    def test_high_access_count_slows_decay(self):
        """Frequently accessed memories should get higher stability."""
        system = AdaptiveDecaySystem()
        base = DecayConfig(stability_seconds=604800)
        now = datetime.now(timezone.utc)

        low_access = _memory(access_count=0, last_accessed=now)
        high_access = _memory(access_count=100, last_accessed=now)

        config_low = system.adjust_decay_config(low_access, base, now)
        config_high = system.adjust_decay_config(high_access, base, now)

        assert config_high.stability_seconds > config_low.stability_seconds

    def test_recent_access_slows_decay(self):
        """Recently accessed memories should decay slower."""
        system = AdaptiveDecaySystem()
        base = DecayConfig(stability_seconds=604800)
        now = datetime.now(timezone.utc)

        recent = _memory(last_accessed=now - timedelta(hours=1))
        old = _memory(last_accessed=now - timedelta(days=60))

        config_recent = system.adjust_decay_config(recent, base, now)
        config_old = system.adjust_decay_config(old, base, now)

        assert config_recent.stability_seconds > config_old.stability_seconds

    def test_high_importance_slows_decay(self):
        system = AdaptiveDecaySystem()
        base = DecayConfig(stability_seconds=604800)
        now = datetime.now(timezone.utc)

        high_imp = _memory(importance=0.9, last_accessed=now)
        low_imp = _memory(importance=0.1, last_accessed=now)

        config_high = system.adjust_decay_config(high_imp, base, now)
        config_low = system.adjust_decay_config(low_imp, base, now)

        assert config_high.stability_seconds > config_low.stability_seconds

    def test_update_importance_decreases_over_time(self):
        system = AdaptiveDecaySystem()
        past = datetime(2024, 1, 1, tzinfo=timezone.utc)
        memory = _memory(importance=0.8, last_accessed=past)
        future = past + timedelta(days=30)
        new_imp = system.update_importance(memory, current_time=future)
        assert new_imp < 0.8


class TestNoveltyAssessment:
    def test_first_memory_is_novel(self):
        system = AdaptiveDecaySystem()
        memory = _memory(embedding=[1.0, 0.0, 0.0])
        level = system.assess_novelty(memory)
        assert level == NoveltyLevel.NEW

    def test_identical_embeddings_become_familiar(self):
        system = AdaptiveDecaySystem(novelty_similarity_threshold=0.99)
        for i in range(3):
            m = _memory(embedding=[1.0, 0.0, 0.0])
            m.memory_id = f"mem-{i}"  # type: ignore[assignment]
            system.assess_novelty(m)

        # Fourth identical should be FAMILIAR or ROUTINE
        m4 = _memory(embedding=[1.0, 0.0, 0.0])
        m4.memory_id = "mem-4"  # type: ignore[assignment]
        level = system.assess_novelty(m4)
        assert level in (NoveltyLevel.FAMILIAR, NoveltyLevel.ROUTINE)


class TestFSRSSchedule:
    def test_again_resets_stability(self):
        schedule = FSRSSchedule(memory_id="m1", stability_seconds=86400)
        schedule.schedule_next_review(grade=1)
        assert schedule.stability_seconds == pytest.approx(60.0)
        assert schedule.repetition_count == 0

    def test_good_grows_stability(self):
        schedule = FSRSSchedule(memory_id="m1", stability_seconds=86400)
        schedule.schedule_next_review(grade=3)
        assert schedule.stability_seconds > 86400

    def test_easy_grows_more_than_good(self):
        s_good = FSRSSchedule(memory_id="m1", stability_seconds=86400, difficulty=5.0)
        s_easy = FSRSSchedule(memory_id="m2", stability_seconds=86400, difficulty=5.0)
        s_good.schedule_next_review(grade=3)
        s_easy.schedule_next_review(grade=4)
        assert s_easy.stability_seconds > s_good.stability_seconds

    def test_stability_capped_at_365_days(self):
        schedule = FSRSSchedule(memory_id="m1", stability_seconds=300 * 86400)
        for _ in range(20):
            schedule.schedule_next_review(grade=4)
        assert schedule.stability_seconds <= 365 * 86400

    def test_next_review_is_set(self):
        schedule = FSRSSchedule(memory_id="m1")
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        schedule.schedule_next_review(grade=3, current_time=now)
        assert schedule.next_review is not None
        assert schedule.next_review > now


class TestSpacedRepetitionIntegration:
    def test_register_and_get_due(self):
        system = AdaptiveDecaySystem()
        memory = _memory()
        system.register_for_spaced_repetition(memory, initial_grade=3)

        future = datetime.now(timezone.utc) + timedelta(days=30)
        due = system.get_due_reviews(current_time=future)
        assert memory.memory_id in due

    def test_record_recall_updates_schedule(self):
        system = AdaptiveDecaySystem()
        memory = _memory()
        system.register_for_spaced_repetition(memory, initial_grade=3)

        initial_stability = system.sr_schedules[memory.memory_id].stability_seconds
        system.record_recall(memory.memory_id, grade=4)
        new_stability = system.sr_schedules[memory.memory_id].stability_seconds
        assert new_stability > initial_stability


class TestFSRSEdgeCases:
    def test_invalid_grade_uses_default_multiplier(self):
        """Invalid grade values (>4) use default multiplier."""
        s = FSRSSchedule(memory_id="m1", stability_seconds=86400.0, difficulty=5.0)
        s.schedule_next_review(grade=99)
        # Default multiplier=2.5, difficulty_factor=1.0 at diff=5.0
        assert s.stability_seconds == pytest.approx(216000.0)
        assert s.repetition_count == 1

    def test_difficulty_clamped_at_boundaries(self):
        """Difficulty stays within [1.0, 10.0] even with extreme grades."""
        s1 = FSRSSchedule(memory_id="m1", difficulty=1.0)
        s1.schedule_next_review(grade=4)  # delta=-0.5
        assert s1.difficulty >= 1.0

        s2 = FSRSSchedule(memory_id="m2", difficulty=9.5)
        s2.schedule_next_review(grade=1)  # delta=+2.0 → would be 11.5
        assert s2.difficulty <= 10.0

    def test_stability_capped_at_365_days(self):
        """Stability never exceeds 365 days."""
        s = FSRSSchedule(
            memory_id="m1",
            stability_seconds=300 * 86400.0,  # 300 days
            difficulty=1.0,  # Low difficulty → high multiplier
        )
        s.schedule_next_review(grade=4)
        assert s.stability_seconds <= 365 * 86400

    def test_grade_1_resets_repetition_count(self):
        """Grade 1 (Again) resets to 0 reps and minimal stability."""
        s = FSRSSchedule(memory_id="m1", repetition_count=10, stability_seconds=86400.0)
        s.schedule_next_review(grade=1)
        assert s.repetition_count == 0
        assert s.stability_seconds == pytest.approx(60.0)


class TestAdaptiveUpdateImportance:
    def test_zero_elapsed_time_minimal_decay(self):
        """Memory just accessed should barely decay."""
        system = AdaptiveDecaySystem()
        now = datetime.now(timezone.utc)
        mem = _memory(importance=0.8, last_accessed=now, access_count=5)
        new_imp = system.update_importance(mem, current_time=now)
        assert new_imp >= 0.7  # Minimal decay

    def test_spaced_repetition_boost_when_due(self):
        """SR boost adds 0.1 when next_review is due."""
        system = AdaptiveDecaySystem(enable_spaced_repetition=True)
        now = datetime.now(timezone.utc)
        mem = _memory(importance=0.5, last_accessed=now, access_count=5)
        system.register_for_spaced_repetition(mem, initial_grade=3)

        # Force next_review to be in the past
        schedule = system.sr_schedules[mem.memory_id]
        schedule.next_review = now - timedelta(seconds=1)

        new_imp = system.update_importance(mem, current_time=now)
        # Should include the 0.1 boost, capped at 1.0
        assert new_imp > 0.5


class TestDueReviews:
    def test_empty_schedules(self):
        """No registered memories → empty list."""
        system = AdaptiveDecaySystem()
        assert system.get_due_reviews() == []

    def test_partial_due(self):
        """Only memories past next_review are returned."""
        system = AdaptiveDecaySystem()
        now = datetime.now(timezone.utc)

        m1 = _memory()
        m1.memory_id = "m1"
        m2 = _memory()
        m2.memory_id = "m2"

        system.register_for_spaced_repetition(m1, initial_grade=3)
        system.register_for_spaced_repetition(m2, initial_grade=3)

        # Make only m1 overdue
        system.sr_schedules["m1"].next_review = now - timedelta(seconds=1)

        due = system.get_due_reviews(current_time=now)
        assert "m1" in due
        assert "m2" not in due


class TestBackwardCompat:
    def test_adaptive_importance_system_alias(self):
        """Backward compat aliases should only exist in experimental."""
        from mnemotree.experimental import AdaptiveImportanceSystem

        assert AdaptiveImportanceSystem is AdaptiveDecaySystem

    def test_spaced_repetition_schedule_alias(self):
        """Backward compat aliases should only exist in experimental."""
        from mnemotree.experimental import SpacedRepetitionSchedule

        assert SpacedRepetitionSchedule is FSRSSchedule

    def test_import_from_experimental(self):
        from mnemotree.experimental import AdaptiveImportanceSystem as AIS
        from mnemotree.experimental import SpacedRepetitionSchedule as SRS

        assert AIS is AdaptiveDecaySystem
        assert SRS is FSRSSchedule
