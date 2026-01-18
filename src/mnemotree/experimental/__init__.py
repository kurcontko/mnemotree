"""
Experimental features for mnemotree.

This module contains advanced features that are useful but still in development:

- **AdaptiveImportanceSystem**: Dynamic decay based on usage patterns, recency, and novelty
- **MemoryConsolidator**: Intelligently merges similar memories to reduce redundancy
- **ClaimsRegistry**: Tracks factual claims and detects conflicts/staleness
- **ContextAwareWriteGate**: Quality filtering before storing new memories

These features are imported by `mnemotree.configs` for pre-configured setups.

Usage:
    from mnemotree.experimental import AdaptiveImportanceSystem, DecayParameters
    from mnemotree.experimental import MemoryConsolidator, ConsolidationConfig
    from mnemotree.experimental import ClaimsRegistry
    from mnemotree.experimental import ContextAwareWriteGate, WritePolicy
"""

from .adaptive_decay import (
    AdaptiveImportanceSystem,
    DecayParameters,
    DecayProfile,
    NoveltyLevel,
    SpacedRepetitionSchedule,
)
from .consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    MemoryConsolidator,
)
from .truth_maintenance import (
    Claim,
    ClaimsRegistry,
    ClaimStatus,
)
from .write_gate import (
    ContextAwareWriteGate,
    WriteDecision,
    WritePolicy,
    WriteResult,
)

__all__ = [
    # Adaptive decay
    "AdaptiveImportanceSystem",
    "DecayParameters",
    "DecayProfile",
    "NoveltyLevel",
    "SpacedRepetitionSchedule",
    # Consolidation
    "ConsolidationConfig",
    "ConsolidationResult",
    "MemoryConsolidator",
    # Truth maintenance
    "Claim",
    "ClaimsRegistry",
    "ClaimStatus",
    # Write gate
    "ContextAwareWriteGate",
    "WriteDecision",
    "WritePolicy",
    "WriteResult",
]
