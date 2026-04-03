"""Content normalization for memory ingestion.

Provides coreference resolution and temporal anchoring to improve
retrieval quality by replacing pronouns with proper names and
relative dates with absolute dates.
"""

from __future__ import annotations

from .base import BaseNormalizer
from .coref import CoreferenceNormalizer
from .pipeline import NormalizationPipeline
from .temporal import TemporalNormalizer


def create_normalization_pipeline(
    *,
    enable_coref: bool = True,
    enable_temporal: bool = True,
    coref_backend: str = "heuristic",
) -> NormalizationPipeline | None:
    """Create a normalization pipeline based on configuration.

    Args:
        enable_coref: Enable coreference resolution.
        enable_temporal: Enable temporal anchoring.
        coref_backend: Coreference backend ("heuristic" or "fastcoref").

    Returns:
        A configured NormalizationPipeline, or None if nothing is enabled.
    """
    normalizers: list[BaseNormalizer] = []

    if enable_coref:
        normalizers.append(CoreferenceNormalizer(backend=coref_backend))
    if enable_temporal:
        normalizers.append(TemporalNormalizer())

    if not normalizers:
        return None

    return NormalizationPipeline(normalizers)


__all__ = [
    "BaseNormalizer",
    "CoreferenceNormalizer",
    "NormalizationPipeline",
    "TemporalNormalizer",
    "create_normalization_pipeline",
]
