from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field

from ..core.models import MemoryType, MemoryCategory, MemoryItem, EmotionCategory


class MemoryAnalysisResult(BaseModel):
    memory_type: MemoryType
    memory_category: MemoryCategory
    importance: float
    emotions: List[str]
    emotional_valence: float
    emotional_arousal: float
    tags: List[str]
    linked_concepts: List[str]
    context_summary: Dict[str, Any]
    

class MemoryClassificationResult(BaseModel):
    memory_type: MemoryType = Field(description=f"One of {[t.value for t in MemoryType]}")
    memory_category: MemoryCategory = Field(description=f"One of {[c.value for c in MemoryCategory]}")
    importance: float = Field(description="Float between 0-1 indicating importance")
    rationale: str = Field(description="Brief explanation of classification")
    
    
class EmotionAnalysisResult(BaseModel):
    emotions: List[str] = Field(description="List of emotions")
    emotional_valence: float = Field(description="Float between -1 (negative) and 1 (positive)")
    emotional_arousal: float = Field(description="Float between 0 (calm) and 1 (excited)")
    rationale: str = Field(description="Brief explanation of emotional analysis")
    
    
class ConceptExtractionResult(BaseModel):
    tags: List[str] = Field(description="List of relevant tags")
    linked_concepts: List[str] = Field(description="List of related concepts")
    context_summary: Dict[str, Any] = Field(description="Important contextual information")
    rationale: str = Field(description="Brief explanation of extracted concepts")