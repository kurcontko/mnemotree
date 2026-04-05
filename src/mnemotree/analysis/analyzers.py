from __future__ import annotations

from .base import BaseAnalyzer
from .models import (
    ConceptExtractionResult,
    EmotionAnalysisResult,
    InsightsResult,
    MemoryClassificationResult,
)


class MemoryClassifierAnalyzer(BaseAnalyzer):
    """Handles memory classification analysis."""

    def _get_output_type(self):
        return MemoryClassificationResult

    def _get_system_prompt(self) -> str:
        return """Analyze the following conversation or content and classify it.

Classify the memory type and importance of the content.
Declarative memory types include:
- "episodic" - Personal experiences
- "semantic" - Facts and general knowledge
- "autobiographical" - Personal life story
- "prospective" - Future intentions

Non-Declarative (Implicit) Memory include:
- "procedural" - Skills and procedures
- "priming" - Influence of prior exposure
- "conditioning" - Learned associations

Short-term processing memory:
- "working" - Short-term processing

Based on rationale, score the importance of the memory between 0 and 1."""


class EmotionAnalyzer(BaseAnalyzer):
    """Handles emotional content analysis."""

    def _get_output_type(self):
        return EmotionAnalysisResult

    def _get_system_prompt(self) -> str:
        return "Analyze the emotional content of the provided text and return structured JSON."


class ConceptAnalyzer(BaseAnalyzer):
    """Handles concept extraction analysis."""

    def _get_output_type(self):
        return ConceptExtractionResult

    def _get_system_prompt(self) -> str:
        return "Analyze the provided content and extract key concepts. Return structured JSON."


class PatternAnalyzer(BaseAnalyzer):
    """Handles pattern and insight analysis across memories."""

    def _get_output_type(self):
        return InsightsResult

    def _get_system_prompt(self) -> str:
        return (
            "Analyze the provided memories and identify key patterns, themes, and insights. "
            "Return structured JSON."
        )
