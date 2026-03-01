"""Tests for mnemotree.core.decay — the unified decay engine."""

from __future__ import annotations

import pytest

from mnemotree.core.decay import (
    DecayConfig,
    ForgettingCurve,
    StabilityUpdater,
    compute_decayed_importance,
)


class TestForgettingCurve:
    def test_retrievability_at_stability_equals_target(self):
        """R(S) should equal target_retention."""
        curve = ForgettingCurve(decay_power=0.5, target_retention=0.9)
        stability = 604800.0  # 7 days
        r = curve.retrievability(stability, stability)
        assert r == pytest.approx(0.9, abs=1e-6)

    def test_retrievability_at_zero_equals_one(self):
        """R(0) = 1 (perfect recall at t=0)."""
        curve = ForgettingCurve()
        assert curve.retrievability(0, 604800) == pytest.approx(1.0)

    def test_retrievability_monotonically_decreasing(self):
        """Retrievability should decrease as elapsed time grows."""
        curve = ForgettingCurve()
        stability = 604800.0
        prev = 1.0
        for hours in [1, 6, 24, 72, 168, 336, 720]:
            r = curve.retrievability(hours * 3600, stability)
            assert r < prev, f"R should decrease at {hours}h"
            assert r > 0, f"R should stay positive at {hours}h"
            prev = r

    def test_retrievability_with_zero_stability(self):
        curve = ForgettingCurve()
        assert curve.retrievability(100, 0) == pytest.approx(0.0)

    def test_different_target_retention(self):
        curve = ForgettingCurve(target_retention=0.8)
        r = curve.retrievability(604800, 604800)
        assert r == pytest.approx(0.8, abs=1e-6)

    def test_different_decay_power(self):
        """Higher decay_power = faster drop."""
        curve_slow = ForgettingCurve(decay_power=0.3)
        curve_fast = ForgettingCurve(decay_power=0.8)
        stability = 604800.0
        elapsed = 2 * stability
        assert curve_slow.retrievability(elapsed, stability) > curve_fast.retrievability(
            elapsed, stability
        )


class TestDecayConfig:
    def test_from_half_life_produces_correct_half_life(self):
        """The half-life point should give R ≈ 0.5."""
        half_life = 3 * 86400  # 3 days
        config = DecayConfig.from_half_life(half_life)
        curve = ForgettingCurve(
            decay_power=config.decay_power,
            target_retention=config.target_retention,
        )
        r = curve.retrievability(half_life, config.stability_seconds)
        assert r == pytest.approx(0.5, abs=0.01)

    def test_defaults(self):
        config = DecayConfig()
        assert config.stability_seconds == pytest.approx(604800.0)
        assert config.decay_power == pytest.approx(0.5)
        assert config.floor == pytest.approx(0.1)
        assert config.target_retention == pytest.approx(0.9)


class TestComputeDecayedImportance:
    def test_no_elapsed_returns_base(self):
        config = DecayConfig()
        result = compute_decayed_importance(0.8, 0.0, config)
        assert result == pytest.approx(0.8)

    def test_decays_over_time(self):
        config = DecayConfig(stability_seconds=86400)
        fresh = compute_decayed_importance(0.8, 0.0, config)
        aged = compute_decayed_importance(0.8, 7 * 86400, config)
        assert aged < fresh

    def test_floor_protection(self):
        config = DecayConfig(stability_seconds=3600, floor=0.2)
        result = compute_decayed_importance(0.8, 365 * 86400, config)
        assert result >= 0.2

    def test_floor_does_not_inflate(self):
        """Floor should never raise importance above its base."""
        config = DecayConfig(floor=0.5)
        result = compute_decayed_importance(0.1, 86400, config)
        assert result <= 0.1

    def test_with_explicit_curve(self):
        curve = ForgettingCurve(decay_power=0.5, target_retention=0.9)
        config = DecayConfig()
        result = compute_decayed_importance(1.0, 604800, config, curve=curve)
        assert result == pytest.approx(0.9, abs=0.01)


class TestStabilityUpdater:
    def test_growth_on_recall(self):
        updater = StabilityUpdater()
        initial = 86400.0  # 1 day
        new = updater.update(initial, retrievability=0.9)
        assert new > initial

    def test_more_growth_when_retrievability_low(self):
        """Low R means harder recall => bigger stability increase."""
        updater = StabilityUpdater()
        stability = 86400.0
        new_easy = updater.update(stability, retrievability=0.95)
        new_hard = updater.update(stability, retrievability=0.5)
        assert new_hard > new_easy

    def test_cap_at_max_stability(self):
        updater = StabilityUpdater(max_stability_seconds=30 * 86400)
        result = updater.update(29 * 86400, retrievability=0.1)
        assert result <= 30 * 86400

    def test_zero_stability_unchanged(self):
        updater = StabilityUpdater()
        assert updater.update(0, 0.5) == pytest.approx(0)


class TestDecayConfigValidation:
    def test_zero_decay_power_raises(self):
        with pytest.raises(ValueError, match="decay_power must be positive"):
            DecayConfig.from_half_life(86400, decay_power=0)

    def test_negative_decay_power_raises(self):
        with pytest.raises(ValueError, match="decay_power must be positive"):
            DecayConfig.from_half_life(86400, decay_power=-0.5)
