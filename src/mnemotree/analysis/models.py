from __future__ import annotations

# memory/models.py
from typing import Any

from pydantic import BaseModel, Field

from ..core.models import (
    EmotionCategory,
    MemoryType,
)


class MemoryAnalysisResult(BaseModel):
    memory_type: MemoryType
    importance: float
    emotional_valence: float | None
    emotional_arousal: float | None
    emotions: list[EmotionCategory] | list[str] | None
    tags: list[str] | None
    linked_concepts: list[str] | None
    context_summary: dict[str, Any] | None


class MemoryClassificationResult(BaseModel):
    memory_type: MemoryType = Field(description=f"One of {[t.value for t in MemoryType]}")
    importance: float = Field(description="Float between 0-1 indicating importance")
    rationale: str = Field(description="Brief explanation of classification")


class EmotionAnalysisResult(BaseModel):
    # emotions: Union[List[EmotionCategory], List[str]] = Field(description=f"List of emotions, possible values: {[e.value for e in EmotionCategory]}")
    emotions: list[str] = Field(
        description=f"List of emotions, possible values: {[e.value for e in EmotionCategory]}"
    )
    emotional_valence: float = Field(description="Float between -1 (negative) and 1 (positive)")
    emotional_arousal: float = Field(description="Float between 0 (calm) and 1 (excited)")
    rationale: str = Field(description="Brief explanation of emotional analysis")


class ConceptExtractionResult(BaseModel):
    tags: list[str] = Field(description="List of relevant tags")
    linked_concepts: list[str] = Field(description="List of related concepts")
    context_summary: dict[str, Any] = Field(description="Important contextual information")
    rationale: str = Field(description="Brief explanation of extracted concepts")


class InsightsResult(BaseModel):
    patterns: list[str] = Field(description="List of detected patterns")
    insights: list[str] = Field(description="List of insights")
    summary: str = Field(description="Summary of insights")
    rationale: str = Field(description="Brief explanation of insights")


class SummaryResult(BaseModel):
    summary: str = Field(description="Concise summary of the interaction")
    topics: list[str] = Field(description="Key topics discussed")
    decisions: list[str] = Field(description="Decisions made, if any")
    action_items: list[str] = Field(description="Action items, if any")
    entities: list[str] = Field(description="Named entities mentioned")
