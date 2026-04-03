"""LocalModelAnalyzer — drop-in replacement for MemoryAnalyzer using local models."""

from __future__ import annotations

import logging
from typing import Any

from ..analysis.models import MemoryAnalysisResult
from ..core.models import MemoryType
from .extractor import ExtractorInference, ExtractorResult
from .router import RouterInference, RouterResult

logger = logging.getLogger(__name__)

# Maps router label strings to MemoryType enum values.
_LABEL_TO_TYPE: dict[str, MemoryType] = {
    "episodic": MemoryType.EPISODIC,
    "semantic": MemoryType.SEMANTIC,
    "procedural": MemoryType.PROCEDURAL,
}


class LocalModelAnalyzer:
    """Analyzer that uses local router + extractor models.

    Provides the same ``async analyze(content, context)`` interface as
    :class:`~mnemotree.analysis.memory_analyzer.MemoryAnalyzer` so it can be
    used as a drop-in replacement in the enrichment pipeline.

    Parameters
    ----------
    router:
        A :class:`RouterInference` instance for classifying memory type.
    extractor:
        A :class:`ExtractorInference` instance for structured extraction.
    """

    def __init__(
        self,
        router: RouterInference,
        extractor: ExtractorInference | None = None,
    ) -> None:
        self.router = router
        self.extractor = extractor

    async def analyze(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> MemoryAnalysisResult:
        """Run router → extractor and return a :class:`MemoryAnalysisResult`.

        The result is compatible with the existing enrichment pipeline:
        ``memory_type``, ``importance``, and ``context_summary`` are populated
        from the local models.  Fields that require an LLM (emotions, tags,
        linked_concepts) are set to empty/neutral defaults.

        If no extractor was provided, only classification is performed and
        importance falls back to the router's highest probability.
        """
        # Stage 1: classify
        router_result: RouterResult = await self.router.classify_record(content, context)
        memory_type = _LABEL_TO_TYPE.get(router_result.primary_type, MemoryType.SEMANTIC)

        logger.debug(
            "Router classified as %s (probs=%s, active=%s)",
            router_result.primary_type,
            router_result.probabilities,
            router_result.active_labels,
        )

        # Build context_summary with router metadata
        context_summary: dict[str, Any] = {
            "router_probabilities": router_result.probabilities,
            "active_labels": router_result.active_labels,
        }

        # Stage 2: extract (if extractor is available)
        if self.extractor is not None:
            extractor_result: ExtractorResult = await self.extractor.extract(
                content,
                router_result.primary_type,
                context,
            )

            logger.debug(
                "Extractor confidence=%.3f parsed=%s",
                extractor_result.confidence,
                extractor_result.parsed is not None,
            )

            importance = max(0.1, min(1.0, extractor_result.confidence))
            if extractor_result.parsed:
                context_summary["extraction"] = extractor_result.parsed
        else:
            # Without extractor, derive importance from router probability.
            importance = max(0.1, router_result.probabilities.get(router_result.primary_type, 0.5))

        return MemoryAnalysisResult(
            memory_type=memory_type,
            importance=importance,
            emotional_valence=None,
            emotional_arousal=None,
            emotions=None,
            tags=None,
            linked_concepts=None,
            context_summary=context_summary,
        )
