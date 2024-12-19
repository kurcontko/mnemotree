# core/models.py
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json
import textwrap

import numpy as np
from pydantic import BaseModel, Field, validator
from langchain.schema import Document


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
    
    # Additional Types
    ENTITIES = "entities"             # Entity extraction results
    
    @property
    def category(self) -> str:
        if self in [self.EPISODIC, self.SEMANTIC, self.AUTOBIOGRAPHICAL, self.PROSPECTIVE]:
            return "declarative"
        elif self in [self.PROCEDURAL, self.PRIMING, self.CONDITIONING]:
            return "non_declarative"
        elif self == self.WORKING:
            return "working"
        else:
            raise ValueError(f"Unknown category for memory type: {self}")


class EmotionCategory(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"
    SATISFACTION = "satisfaction"
    EXCITEMENT = "excitement"


class MemoryItem(BaseModel):
    """A flattened representation of a memory item optimized for vector and graph database operations.
    
    This class uses a deliberately flattened structure to optimize for:
    1. Vector database storage and retrieval
    2. Graph database querying efficiency
    3. Simplified serialization/deserialization
    4. Easier migration between different database systems
    
    Rather than using nested objects, relationships are represented through:
    - Direct fields (e.g., emotional_valence instead of an Emotion object)
    - Reference IDs (e.g., previous_event_id instead of nested Event objects)
    - Lists of references (e.g., linked_concepts as string IDs)
    
    Attributes:
        memory_id (str): Unique identifier for the memory
        conversation_id (Optional[str]): Reference to parent conversation
        user_id (Optional[str]): Reference to owner/creator
        
        # Core Information
        content (str): Main content of the memory
        summary (Optional[str]): Condensed version of content
        tags (Optional[List[str]]): Categorization labels
        author (Optional[str]): Creator of the memory
        memory_type (MemoryType): Type classification (episodic, semantic, etc.)
        timestamp (str): Creation time in UTC
        
        # Access Tracking
        last_accessed (str): Last retrieval time in UTC
        access_count (int): Number of times retrieved
        access_history (List[str]): Timestamp history of accesses
        
        # Quality Metrics
        importance (float): Relevance score (0-1)
        decay_rate (float): Memory degradation rate
        confidence (float): Certainty level (0-1)
        fidelity (float): Quality/accuracy score (0-1)
        
        # Emotional Components (flattened from EmotionalContext)
        emotional_valence (Optional[float]): Negative to positive (-1 to 1)
        emotional_arousal (Optional[float]): Intensity level (0-1)
        emotions (List[Union[EmotionCategory, str]]): Identified emotions
        
        # Relationships (flattened from Connections)
        linked_concepts (Optional[List[str]]): Related concept IDs
        associations (List[str]): Positively related memory IDs
        conflicts_with (List[str]): Negatively related memory IDs
        previous_event_id (Optional[str]): Temporal predecessor
        next_event_id (Optional[str]): Temporal successor
        
        # Source Attribution (flattened from SourceInfo)
        source (Optional[str]): Origin reference
        credibility (Optional[float]): Source reliability (0-1)
        
        # Vector Representation
        embedding (Optional[List[float]]): Vector embedding for similarity search
        
        # Additional Data
        context (Optional[Union[Dict[str, Any], str]]): Contextual information
        metadata (Dict[str, Any]): Flexible additional attributes
    """
    # Core Identifiers
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[str] = None # TODO: Handle this field in memory core
    user_id: Optional[str] = None # TODO: Handle this field in memory core
    
    # Core Information
    content: str
    summary: Optional[str] = ""
    tags: Optional[List[str]] = Field(default_factory=list)
    
    entities: Optional[Dict[str, str]] =  Field(default_factory=dict) # entity text -> entity type mapping
    entity_mentions: Optional[Dict[str, List[str]]] =  Field(default_factory=dict) # entity -> contexts where it appears
    author: Optional[str] = ""
    memory_type: MemoryType
    timestamp: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))
    
    # Access information
    last_accessed: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))
    access_count: int = 0
    access_history: List[str] = Field(default_factory=list)
    
    # Metrics
    importance: float  # Should be between 0 and 1
    decay_rate: float = 0.01
    confidence: float = 1.0
    fidelity: float = 1.0
    
    # Emotional Analysis - # TODO: Consider using a separate EmotionalContext class, currently flattened
    emotional_valence: Optional[float] = Field(None, ge=-1.0, le=1.0)  # -1 to 1
    emotional_arousal: Optional[float] = Field(None, ge=0.0, le=1.0)  # 0 to 1
    #emotions: Optional[Union[List[EmotionCategory], List[str]]] = Field(default_factory=list)
    emotions: Optional[List[str]] = Field(default_factory=list)
    
    # Connections - # TODO: Consider using a separate Connections class, currently flattened
    linked_concepts: Optional[List[str]] = Field(default_factory=list)
    associations: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    previous_event_id: Optional[str] = None
    next_event_id: Optional[str] = None
    
    # Source Information - # TODO: Consider using a separate SourceInfo class, currently flattened
    source: Optional[str] = "conversation"  # Default to conversation
    credibility: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Embeddings
    embedding: Optional[List[float]] = None # TODO: Consider using numpy array
    
    # Metadata
    context: Optional[Union[Dict[str, Any], str]] = Field(default_factory=dict) # TODO: Refactor this field
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('importance')
    def importance_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('importance must be between 0 and 1')
        return v

    def update_access(self):
        self.access_count += 1
        self.last_accessed = str(datetime.now(timezone.utc))
        self.access_history.append(self.last_accessed)
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
        Creates a concise, formatted string representation of the MemoryItem.
        """
        def format_time(time_str: str) -> str:
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M UTC')
            except:
                return time_str

        def format_float(value: float) -> str:
            return f"{value:.2f}" if value is not None else "N/A"

        def get_rating(value: float) -> str:
            if value is None:
                return ""
            filled = round(value * 5)
            return f"[{'|' * filled}{'-' * (5 - filled)}]"

        # Core Information
        output = [f"### Memory: {self.memory_id}"]
        
        if self.content:
            output.append(f"```\n{self.content}\n```")
        
        if self.summary:
            output.append(f"**Summary:** {self.summary}")

        # Key Details
        details = [
            f"**Type:** {self.memory_type}",
            f"**Created:** {format_time(self.timestamp)}",
        ]
        if self.tags:
            details.append(f"**Tags:** {', '.join(self.tags)}")
        output.append(" | ".join(details))

        # Metrics (single line)
        metrics = []
        if self.importance is not None:
            metrics.append(f"**Imp:** {format_float(self.importance)} {get_rating(self.importance)}")
        if self.confidence is not None:
            metrics.append(f"**Conf:** {format_float(self.confidence)} {get_rating(self.confidence)}")
        if self.fidelity is not None:
            metrics.append(f"**Fid:** {format_float(self.fidelity)} {get_rating(self.fidelity)}")
        if metrics:
            output.append(" | ".join(metrics))

        # Emotional Context (if present)
        emotional = []
        if self.emotional_valence is not None or self.emotional_arousal is not None:
            if self.emotional_valence is not None:
                sentiment = "(+)" if self.emotional_valence > 0 else "(-)" if self.emotional_valence < 0 else "(=)"
                emotional.append(f"**Val:** {format_float(self.emotional_valence)}{sentiment}")
            if self.emotional_arousal is not None:
                emotional.append(f"**Aro:** {format_float(self.emotional_arousal)}")
            if self.emotions:
                emotional.append(f"**Emo:** {', '.join(self.emotions)}")
            output.append(" | ".join(emotional))

        # Connections (if present)
        if any([self.associations, self.linked_concepts, self.conflicts_with]):
            connections = []
            if self.associations:
                connections.append(f"**Assoc:** {', '.join(self.associations)}")
            if self.linked_concepts:
                connections.append(f"**Links:** {', '.join(self.linked_concepts)}")
            if self.conflicts_with:
                connections.append(f"**Conflicts:** {', '.join(self.conflicts_with)}")
            output.append(" | ".join(connections))

        # Timeline (if present)
        if self.previous_event_id or self.next_event_id:
            timeline = []
            if self.previous_event_id:
                timeline.append(f"← {self.previous_event_id}")
            if self.next_event_id:
                timeline.append(f"{self.next_event_id} →")
            output.append("**Timeline:** " + " | ".join(timeline))

        # Source and Credibility (if present)
        if self.source or self.credibility is not None:
            source_info = []
            if self.source:
                source_info.append(f"**Source:** {self.source}")
            if self.credibility is not None:
                source_info.append(f"**Cred:** {format_float(self.credibility)} {get_rating(self.credibility)}")
            output.append(" | ".join(source_info))

        # Additional Data (if present, in collapsed form)
        if self.context or self.metadata:
            if self.context:
                output.append(f"**Context:** ```{json.dumps(self.context)}```")
            if self.metadata:
                output.append(f"**Metadata:** ```{json.dumps(self.metadata)}```")

        return "\n".join(output)
    
    def to_str_llm(self) -> str:
        """
        Creates a simplified string representation optimized for LLM consumption.
        Focuses on core information and critical context while maintaining a clean format.
        """
        parts = []
        
        # Core information with essential context
        parts.append(f"Memory ({self.memory_type.value}):")
        parts.append(f"Content: {self.content}")
        
        if self.summary:
            parts.append(f"Summary: {self.summary}")
            
        # Important metadata that might influence LLM understanding
        parts.append(f"Importance: {self.importance:.2f}")
        
        # Emotional context if available
        if any([self.emotional_valence is not None, 
                self.emotional_arousal is not None, 
                self.emotions]):
            emotion_parts = []
            if self.emotional_valence is not None:
                emotion_parts.append(f"valence: {self.emotional_valence:+.2f}")
            if self.emotional_arousal is not None:
                emotion_parts.append(f"arousal: {self.emotional_arousal:.2f}")
            if self.emotions:
                emotion_parts.append(f"emotions: {', '.join(self.emotions)}")
            parts.append(f"Emotional Context: {' | '.join(emotion_parts)}")
        
        # Critical relationships
        if self.linked_concepts:
            parts.append(f"Related Concepts: {', '.join(self.linked_concepts)}")
            
        # Entity information if available
        if self.entities:
            parts.append(f"Entities: {', '.join([f'{k} ({v})' for k, v in self.entities.items()])}")
        
        # Temporal context if available
        if self.previous_event_id or self.next_event_id:
            timeline = []
            if self.previous_event_id:
                timeline.append(f"previous: {self.previous_event_id}")
            if self.next_event_id:
                timeline.append(f"next: {self.next_event_id}")
            parts.append(f"Timeline: {' | '.join(timeline)}")
        
        return "\n".join(parts)
    
    def to_langchain_document(self) -> "Document":
        """
        Convert the MemoryItem to a LangChain Document.
        Requires langchain to be installed.
        
        Returns:
            Document: A LangChain Document containing the memory content and metadata
        """
        try:
            from langchain.schema import Document
        except ImportError:
            raise ImportError(
                "langchain package is required to use this method. "
                "Please install it with `pip install langchain`"
            )
        
        # Prepare metadata dictionary with all relevant fields
        metadata = {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "memory_category": self.memory_type.category,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "confidence": self.confidence,
            "fidelity": self.fidelity,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            # Include emotional context if present
            "emotional_valence": self.emotional_valence if self.emotional_valence else None,
            "emotional_arousal": self.emotional_arousal if self.emotional_arousal else None,
            "emotions": self.emotions if self.emotions else None,
            # Include connections
            "tags": self.tags,
            "associations": self.associations if self.associations else None,
            "linked_concepts": self.linked_concepts if self.linked_concepts else None,
            # Include source info
            "source": self.source if self.source else None,
        }
        
        # Add any custom context and metadata
        metadata.update(self.metadata)
        
        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return Document(
            page_content=self.content,
            metadata=metadata
        )
