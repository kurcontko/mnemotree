"""
Unified decay engine for mnemotree.

Implements FSRS-4.5-style power-law forgetting curve as a single source of truth
for all decay math. Pure functions, no side effects.

References:
- FSRS-4.5: https://github.com/open-spaced-repetition/fsrs4anki
- FadeMem: shape parameter approach for per-type decay profiles
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import MemoryType

# ---------------------------------------------------------------------------
# Forgetting curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgettingCurve:
    """FSRS-4.5 power-law forgetting curve.

    R(t, S) = (1 + factor * t / S) ^ (-decay_power)

    where *factor* is derived so that R(S) == target_retention.
    """

    decay_power: float = 0.5
    target_retention: float = 0.9

    def __post_init__(self) -> None:
        if self.decay_power <= 0:
            raise ValueError("decay_power must be positive")
        if not (0 < self.target_retention < 1):
            raise ValueError("target_retention must be between 0 and 1 (exclusive)")

    @property
    def factor(self) -> float:
        """Derive factor so R(S) = target_retention."""
        # R(S) = (1 + factor)^(-decay_power) = target_retention
        # => 1 + factor = target_retention^(-1/decay_power)
        # => factor = target_retention^(-1/decay_power) - 1
        return self.target_retention ** (-1.0 / self.decay_power) - 1.0

    def retrievability(self, elapsed_seconds: float, stability_seconds: float) -> float:
        """Compute retrievability R in [0, 1].

        Args:
            elapsed_seconds: Time since last access/review.
            stability_seconds: Memory stability (time at which R = target_retention).

        Returns:
            Retrievability score between 0 and 1.
        """
        if stability_seconds <= 0:
            return 0.0
        if elapsed_seconds <= 0:
            return 1.0
        return (1.0 + self.factor * elapsed_seconds / stability_seconds) ** (-self.decay_power)


# ---------------------------------------------------------------------------
# Decay configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for a single decay profile."""

    stability_seconds: float = 604800.0  # 7 days
    decay_power: float = 0.5
    floor: float = 0.1
    target_retention: float = 0.9

    @classmethod
    def from_half_life(cls, half_life_seconds: float, **kwargs) -> DecayConfig:
        """Create a DecayConfig from a half-life value.

        The half-life is the time at which retrievability drops to 0.5.
        We compute stability_seconds so that R(half_life) = 0.5 given the
        power-law model.
        """
        decay_power = kwargs.get("decay_power", 0.5)
        target_retention = kwargs.get("target_retention", 0.9)

        if decay_power <= 0:
            raise ValueError("decay_power must be positive")

        # R(t, S) = (1 + factor * t / S) ^ (-decay_power) = 0.5
        # factor = target_retention^(-1/decay_power) - 1
        factor = target_retention ** (-1.0 / decay_power) - 1.0
        # (1 + factor * half_life / S)^(-decay_power) = 0.5
        # 1 + factor * half_life / S = 0.5^(-1/decay_power)
        # S = factor * half_life / (0.5^(-1/decay_power) - 1)
        denominator = 0.5 ** (-1.0 / decay_power) - 1.0
        if denominator <= 0:
            stability_seconds = half_life_seconds
        else:
            stability_seconds = factor * half_life_seconds / denominator

        return cls(
            stability_seconds=stability_seconds,
            decay_power=decay_power,
            floor=kwargs.get("floor", 0.1),
            target_retention=target_retention,
        )


# ---------------------------------------------------------------------------
# Per-type defaults
# ---------------------------------------------------------------------------

MEMORY_TYPE_DEFAULTS: dict[MemoryType, DecayConfig] = {
    MemoryType.SEMANTIC: DecayConfig(
        stability_seconds=30 * 86400,  # 30 days
        decay_power=0.4,
    ),
    MemoryType.EPISODIC: DecayConfig(
        stability_seconds=7 * 86400,  # 7 days
        decay_power=0.5,
    ),
    MemoryType.WORKING: DecayConfig(
        stability_seconds=3600,  # 1 hour
        decay_power=0.7,
    ),
    MemoryType.PROCEDURAL: DecayConfig(
        stability_seconds=60 * 86400,  # 60 days
        decay_power=0.35,
    ),
    MemoryType.AUTOBIOGRAPHICAL: DecayConfig(
        stability_seconds=14 * 86400,  # 14 days
        decay_power=0.45,
    ),
    MemoryType.PROSPECTIVE: DecayConfig(
        stability_seconds=3 * 86400,  # 3 days
        decay_power=0.6,
    ),
}

_DEFAULT_CONFIG = DecayConfig()


# ---------------------------------------------------------------------------
# Core decay function
# ---------------------------------------------------------------------------


def compute_decayed_importance(
    base_importance: float,
    elapsed_seconds: float,
    config: DecayConfig,
    curve: ForgettingCurve | None = None,
) -> float:
    """Compute decayed importance using the power-law forgetting curve.

    Args:
        base_importance: Original importance value in [0, 1].
        elapsed_seconds: Time since last access in seconds.
        config: Decay configuration to use.
        curve: Optional pre-built ForgettingCurve (created from config if None).

    Returns:
        Decayed importance, floored but never inflated above base_importance.
    """
    if curve is None:
        curve = ForgettingCurve(
            decay_power=config.decay_power,
            target_retention=config.target_retention,
        )

    r = curve.retrievability(elapsed_seconds, config.stability_seconds)
    decayed = base_importance * r

    # Floor should never exceed base_importance (prevent inflation)
    effective_floor = min(config.floor, base_importance)
    return max(effective_floor, decayed)


# ---------------------------------------------------------------------------
# Stability updater (FSRS-style growth on successful recall)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityUpdater:
    """Simplified FSRS stability growth on successful recall.

    Formula: S' = S * (1 + growth_rate * S^(-shape) * (exp(sensitivity*(1-R)) - 1))

    Capped at max_stability_seconds.
    """

    growth_rate: float = 1.0
    shape: float = 0.5
    sensitivity: float = 5.0
    max_stability_seconds: float = 365 * 86400  # 365 days

    def update(self, current_stability: float, retrievability: float) -> float:
        """Compute new stability after successful recall.

        Args:
            current_stability: Current stability in seconds.
            retrievability: Retrievability at the time of recall (0-1).

        Returns:
            Updated stability in seconds, capped at max_stability_seconds.
        """
        if current_stability <= 0:
            return 0.0

        exponent = self.sensitivity * (1.0 - retrievability)
        # Clamp exponent to avoid overflow
        exponent = min(exponent, 20.0)

        growth = self.growth_rate * current_stability ** (-self.shape) * (math.exp(exponent) - 1.0)
        new_stability = current_stability * (1.0 + growth)
        return min(new_stability, self.max_stability_seconds)
