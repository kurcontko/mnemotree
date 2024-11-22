from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json

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
    
    
class EmotionCategory(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class MemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    author: Optional[str] = ""
    memory_category: MemoryCategory
    memory_type: MemoryType
    timestamp: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))
    importance: float  # Should be between 0 and 1
    decay_rate: float = 0.01
    last_accessed: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))
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
        self.last_accessed = str(datetime.now(timezone.utc))
        self.importance = min(1.0, self.importance + self.decay_rate)  # Reinforce importance upon access

    def decay_importance(self, current_time: datetime):
       # Parse the string back into a datetime object
        last_accessed_dt = datetime.strptime(self.last_accessed, "%Y-%m-%d %H:%M:%S.%f%z")
        
        # Calculate the time difference in seconds
        time_diff = (current_time - last_accessed_dt).total_seconds()
        decay_amount = self.decay_rate * time_diff
        self.importance = max(0, self.importance - decay_amount)

    def to_str(self) -> str:
        """
        Creates a formatted string representation of the MemoryItem.
        Organizes information into logical sections with clear formatting.
        """
        def format_time(time_str: str) -> str:
            """Format datetime string to be more readable"""
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            except:
                return time_str

        # Core Information Section
        core_info = [
            "📝 MEMORY ITEM",
            f"ID: {self.memory_id}",
            f"Content: {self.content}",
            f"Category: {self.memory_category}",
            f"Type: {self.memory_type}",
            f"Created: {format_time(self.timestamp)}",
        ]
        if self.author:
            core_info.append(f"Author: {self.author}")

        # Metrics Section
        metrics = [
            "\n📊 METRICS",
            f"Importance: {self.importance:.2f}",
            f"Confidence: {self.confidence:.2f}",
            f"Fidelity: {self.fidelity:.2f}",
            f"Access Count: {self.access_count}",
            f"Last Accessed: {format_time(self.last_accessed)}",
            f"Decay Rate: {self.decay_rate:.3f}",
        ]

        # Emotional Analysis Section
        emotional = ["\n🎭 EMOTIONAL CONTEXT"]
        if self.emotional_valence is not None:
            emotional.append(f"Valence: {self.emotional_valence:+.2f}")  # Use + to show sign
        if self.emotional_arousal is not None:
            emotional.append(f"Arousal: {self.emotional_arousal:.2f}")
        if self.emotions:
            emotional.append(f"Emotions: {', '.join(self.emotions)}")

        # Connections Section
        connections = ["\n🔗 CONNECTIONS"]
        if self.tags:
            connections.append(f"Tags: {', '.join(self.tags)}")
        if self.associations:
            connections.append(f"Associations: {', '.join(self.associations)}")
        if self.linked_concepts:
            connections.append(f"Linked Concepts: {', '.join(self.linked_concepts)}")
        if self.conflicts_with:
            connections.append(f"Conflicts: {', '.join(self.conflicts_with)}")
        if self.previous_event_id or self.next_event_id:
            timeline = []
            if self.previous_event_id:
                timeline.append(f"Previous: {self.previous_event_id}")
            if self.next_event_id:
                timeline.append(f"Next: {self.next_event_id}")
            connections.append(" | ".join(timeline))

        # Source Information Section
        source_info = []
        if self.source or self.source_credibility is not None:
            source_info = ["\n📚 SOURCE"]
            if self.source:
                source_info.append(f"Source: {self.source}")
            if self.source_credibility is not None:
                source_info.append(f"Credibility: {self.source_credibility:.2f}")

        # Additional Data Section
        additional = []
        if self.context or self.metadata or self.sensory_data:
            additional = ["\n📦 ADDITIONAL DATA"]
            if self.context:
                additional.append(f"Context: {json.dumps(self.context, indent=2)}")
            if self.metadata:
                additional.append(f"Metadata: {json.dumps(self.metadata, indent=2)}")
            if self.sensory_data:
                additional.append(f"Sensory Data: {json.dumps(self.sensory_data, indent=2)}")

        # Combine all sections and filter out empty ones
        sections = [
            '\n'.join(core_info),
            '\n'.join(metrics),
            '\n'.join(emotional) if len(emotional) > 1 else None,
            '\n'.join(connections) if len(connections) > 1 else None,
            '\n'.join(source_info) if source_info else None,
            '\n'.join(additional) if additional else None,
        ]
        
        # Join all non-empty sections with double newlines
        return '\n\n'.join(section for section in sections if section is not None)