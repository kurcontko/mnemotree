"""
Backward-compatibility shim.

All classes have moved to ``mnemotree.core.adaptive``.
This module re-exports them so existing imports keep working.
"""

from ..core.adaptive import (
    AdaptiveDecaySystem,
    DecayParameters,
    DecayProfile,
    FSRSSchedule,
    NoveltyLevel,
)

# Deprecated aliases (kept for backward compat)
AdaptiveImportanceSystem = AdaptiveDecaySystem
SpacedRepetitionSchedule = FSRSSchedule

__all__ = [
    "AdaptiveDecaySystem",
    "AdaptiveImportanceSystem",
    "DecayParameters",
    "DecayProfile",
    "FSRSSchedule",
    "NoveltyLevel",
    "SpacedRepetitionSchedule",
]
