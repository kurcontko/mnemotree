from __future__ import annotations

# memory/models.py
from typing import Dict, List, Optional, Tuple, Any, Union

from pydantic import BaseModel, Field

from ..core.models import (
    MemoryType, 
    MemoryItem,  
    EmotionCategory, 
)


class MemoryAnalysisResult(BaseModel):
    memory_type: MemoryType
    importance: float
    emotional_valence: Optional[float]
    emotional_arousal: Optional[float]
    emotions: Optional[Union[List[EmotionCategory], List[str]]]
    tags: Optional[List[str]]
    linked_concepts: Optional[List[str]]
    context_summary: Optional[Dict[str, Any]]


class MemoryClassificationResult(BaseModel):
    memory_type: MemoryType = Field(description=f"One of {[t.value for t in MemoryType]}")
    importance: float = Field(description="Float between 0-1 indicating importance")
    rationale: str = Field(description="Brief explanation of classification")

    
class EmotionAnalysisResult(BaseModel):
    #emotions: Union[List[EmotionCategory], List[str]] = Field(description=f"List of emotions, possible values: {[e.value for e in EmotionCategory]}")
    emotions: List[str] = Field(description=f"List of emotions, possible values: {[e.value for e in EmotionCategory]}")
    emotional_valence: float = Field(description="Float between -1 (negative) and 1 (positive)")
    emotional_arousal: float = Field(description="Float between 0 (calm) and 1 (excited)")
    rationale: str = Field(description="Brief explanation of emotional analysis")

    
class ConceptExtractionResult(BaseModel):
    tags: List[str] = Field(description="List of relevant tags")
    linked_concepts: List[str] = Field(description="List of related concepts")
    context_summary: Dict[str, Any] = Field(description="Important contextual information")
    rationale: str = Field(description="Brief explanation of extracted concepts")


class InsightsResult(BaseModel):
    patterns: List[str] = Field(description="List of detected patterns")
    insights: List[str] = Field(description="List of insights")
    summary: str = Field(description="Summary of insights")
    rationale: str = Field(description="Brief explanation of insights")


class SummaryResult(BaseModel):
    summary: str = Field(description="Concise summary of the interaction")
    topics: List[str] = Field(description="Key topics discussed")
    decisions: List[str] = Field(description="Decisions made, if any")
    action_items: List[str] = Field(description="Action items, if any")
    entities: List[str] = Field(description="Named entities mentioned")
