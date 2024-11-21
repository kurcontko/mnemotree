from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, validator


class MemoryType(Enum):
    # Declarative (Explicit) Memory
    EPISODIC = "episodic"             # Personal experiences
    SEMANTIC = "semantic"             # Facts and general knowledge
    AUTOBIOGRAPHICAL = "autobiographical"  # Personal life story
    PROSPECTIVE = "prospective"       # Future intentions

    # Non-Declarative (Implicit) Memory
    PROCEDURAL = "procedural"         # Skills and procedures
    PRIMING = "priming"               # Influence of prior exposure
    CONDITIONING = "conditioning"     # Learned associations

    # Working Memory
    WORKING = "working"               # Short-term processing
    
    
class MemoryCategory(Enum):
    DECLARATIVE = "declarative"
    NON_DECLARATIVE = "non_declarative"
    WORKING = "working"


class MemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    memory_category: MemoryCategory
    memory_type: MemoryType
    timestamp: datetime
    importance: float  # Should be between 0 and 1
    decay_rate: float = 0.01
    last_accessed: datetime = Field(default_factory=datetime.now(datetime.timezone.utc))
    access_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    emotional_valence: Optional[float] = None  # -1 (negative) to 1 (positive)
    emotional_arousal: Optional[float] = None  # 0 (calm) to 1 (excited)
    emotions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    associations: List[str] = Field(default_factory=list)
    previous_event_id: Optional[str] = None
    next_event_id: Optional[str] = None
    source: Optional[str] = None
    source_credibility: Optional[float] = None
    confidence: float = 1.0
    fidelity: float = 1.0
    embedding: Optional[List[float]] = None  # Adjust type as needed
    sensory_data: Dict[str, Any] = Field(default_factory=dict)
    linked_concepts: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('importance')
    def importance_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('importance must be between 0 and 1')
        return v

    def update_access(self):
        self.access_count += 1
        self.last_accessed = datetime.now(datetime.timezone.utc)
        self.importance = min(1.0, self.importance + self.decay_rate)  # Reinforce importance upon access

    def decay_importance(self, current_time: datetime):
        time_diff = (current_time - self.last_accessed).total_seconds()
        decay_amount = self.decay_rate * time_diff
        self.importance = max(0, self.importance - decay_amount)
