# core/models.py
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, overload
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class MemoryType(Enum):
    # Declarative (Explicit) Memory
    EPISODIC = "episodic"  # Personal experiences
    SEMANTIC = "semantic"  # Facts and general knowledge
    AUTOBIOGRAPHICAL = "autobiographical"  # Personal life story
    PROSPECTIVE = "prospective"  # Future intentions

    # Non-Declarative (Implicit) Memory
    PROCEDURAL = "procedural"  # Skills and procedures
    PRIMING = "priming"  # Influence of prior exposure
    CONDITIONING = "conditioning"  # Learned associations

    # Working Memory
    WORKING = "working"  # Short-term processing

    # Additional Types
    ENTITIES = "entities"  # Entity extraction results

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

    @property
    def is_episodic(self) -> bool:
        """True for episodic/autobiographical (personal experience) memories."""
        return self in [self.EPISODIC, self.AUTOBIOGRAPHICAL]

    @property
    def is_semantic(self) -> bool:
        """True for semantic (factual knowledge) memories."""
        return self == self.SEMANTIC

    @property
    def is_procedural(self) -> bool:
        """True for procedural (skill/habit) memories."""
        return self in [self.PROCEDURAL, self.PRIMING, self.CONDITIONING]


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


class LinkType(str, Enum):
    """Semantic relationship types inspired by Zettelkasten."""

    SUPPORTS = "supports"  # Evidence for
    CONTRADICTS = "contradicts"  # Evidence against
    ELABORATES = "elaborates"  # Expands concept
    REFERENCES = "references"  # Generic citation
    SIMILAR_TO = "similar_to"  # Semantic similarity
    EXEMPLIFIES = "exemplifies"  # Concrete example
    GENERALIZES = "generalizes"  # Abstract pattern
    CAUSES = "causes"  # Causal relationship
    FOLLOWS = "follows"  # Sequence
    PART_OF = "part_of"  # Component
    DERIVES_FROM = "derives_from"  # Intellectual lineage
    SUPERSEDES = "supersedes"  # Replaces an older memory (A-MEM evolution)
    UPDATES = "updates"  # Partial update of an older memory
    SEQUENCE = "sequence"  # Temporal ordering (MAGMA four-graph)


@overload
def coerce_datetime(value: datetime | str | None, default: datetime) -> datetime: ...


@overload
def coerce_datetime(
    value: datetime | str | None,
    default: datetime | None = None,
) -> datetime | None: ...


def coerce_datetime(
    value: datetime | str | None,
    default: datetime | None = None,
) -> datetime | None:
    dt: datetime | None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f%z")
            except ValueError:
                dt = default
    else:
        dt = default

    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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
        memory_id: Unique identifier for the memory
        conversation_id: Reference to parent conversation
        user_id: Reference to owner/creator
        repo_id: Repository scope identifier for shared agent memory
        worktree_id: Worktree scope identifier within a repository
        task_id: Task scope identifier within a repository/worktree
        agent_id: Agent identity for shared memory and coordination
        run_id: Specific run/session identifier for an agent
        content: Main content of the memory
        summary: Condensed version of content
        tags: Categorization labels
        author: Creator of the memory
        memory_type: Type classification (episodic, semantic, etc.)
        timestamp: Creation time in UTC (storage time)
        event_time: When the event actually happened (vs timestamp = storage time)
        valid_from: Temporal validity start (bi-temporal)
        valid_until: Temporal validity end (None = still valid)
        observation_date: Mastra-style 3-date anchoring — when the observation was made
        referenced_date: Date the memory refers to
        temporal_offset: Relative offset hint ("2 days ago")
        contextual_intent: STITCH inferred intent at ingest time
        is_hot: Codified Context — HOT memories always included, COLD retrieved on-demand
        last_accessed: Last retrieval time in UTC
        access_count: Number of times retrieved
        access_history: Timestamp history of accesses
        importance: Relevance score (0-1)
        decay_rate: Memory degradation rate (deprecated, kept for compat)
        confidence: Certainty level (0-1)
        fidelity: Quality/accuracy score (0-1)
        emotional_valence: Negative to positive (-1 to 1)
        emotional_arousal: Intensity level (0-1)
        emotions: Identified emotions
        linked_concepts: Related concept IDs
        associations: Positively related memory IDs
        conflicts_with: Negatively related memory IDs (auto-populated by conflict detection)
        previous_event_id: Temporal predecessor
        next_event_id: Temporal successor
        source: Origin reference
        credibility: Source reliability (0-1)
        embedding: Vector embedding for similarity search
        context: Contextual information
        metadata: Flexible additional attributes
    """

    # Core Identifiers
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str | None = None  # TODO: Handle this field in memory core
    user_id: str | None = None  # TODO: Handle this field in memory core
    repo_id: str | None = None
    worktree_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    # Core Information
    content: str
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)

    entities: dict[str, str] = Field(default_factory=dict)  # entity text -> entity type mapping
    entity_mentions: dict[str, list[str]] = Field(
        default_factory=dict
    )  # entity -> contexts where it appears
    author: str | None = None
    memory_type: MemoryType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Bi-temporal fields (TSM / Mastra OM 3-date anchoring)
    event_time: datetime | None = None  # When the event actually happened (vs timestamp = storage time)
    valid_from: datetime | None = None  # Temporal validity start
    valid_until: datetime | None = None  # Temporal validity end (None = still valid)
    observation_date: datetime | None = None  # Mastra-style: when the observation was made
    referenced_date: datetime | None = None  # Date the memory refers to
    temporal_offset: str | None = None  # Relative offset hint ("2 days ago", "last week")

    # Retrieval hints
    contextual_intent: str | None = None  # STITCH: inferred intent at ingest time (+35.6% retrieval)
    is_hot: bool = False  # Codified Context: HOT memories always included, COLD retrieved on-demand

    # Access information
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    access_history: list[datetime] = Field(default_factory=list)

    # Metrics
    importance: float  # Should be between 0 and 1
    decay_rate: float = 0.01  # Deprecated: kept for DB backward compat, not used in new decay code
    stability_seconds: float | None = (
        None  # Per-instance stability override (None = use type default)
    )
    reinforcement_rate: float = 0.05  # Controls importance boost on access (separate from decay)
    confidence: float = 1.0
    fidelity: float = 1.0

    # Emotional Analysis - # TODO: Consider using a separate EmotionalContext class, currently flattened
    emotional_valence: float | None = Field(None, ge=-1.0, le=1.0)  # -1 to 1
    emotional_arousal: float | None = Field(None, ge=0.0, le=1.0)  # 0 to 1
    emotions: list[str] = Field(default_factory=list)

    # Connections - # TODO: Consider using a separate Connections class, currently flattened
    linked_concepts: list[str] = Field(default_factory=list)
    associations: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    previous_event_id: str | None = None
    next_event_id: str | None = None

    # Source Information - # TODO: Consider using a separate SourceInfo class, currently flattened
    source: str | None = "conversation"  # Default to conversation
    credibility: float | None = Field(None, ge=0.0, le=1.0)

    # Embeddings
    embedding: list[float] | None = None  # TODO: Consider using numpy array

    # Metadata
    context: dict[str, Any] | str | None = Field(default_factory=dict)  # TODO: Refactor this field
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("importance")
    @classmethod
    def importance_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("importance must be between 0 and 1")
        return v

    def update_access(self, retrievability: float | None = None):
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        self.access_history.append(self.last_accessed)
        self.importance = min(
            1.0, self.importance + self.reinforcement_rate
        )  # Reinforce importance upon access

        # Grow stability on successful recall when per-instance stability is set
        if retrievability is not None and self.stability_seconds is not None:
            from .decay import StabilityUpdater

            updater = StabilityUpdater()
            self.stability_seconds = updater.update(self.stability_seconds, retrievability)

    def decay_importance(self, current_time: datetime):
        from .decay import MEMORY_TYPE_DEFAULTS, DecayConfig, compute_decayed_importance

        last_accessed_dt = coerce_datetime(self.last_accessed, default=current_time)
        elapsed = max(0.0, (current_time - last_accessed_dt).total_seconds())

        config = MEMORY_TYPE_DEFAULTS.get(self.memory_type, DecayConfig())
        if self.stability_seconds is not None:
            config = DecayConfig(
                stability_seconds=self.stability_seconds,
                decay_power=config.decay_power,
                floor=config.floor,
                target_retention=config.target_retention,
            )

        self.importance = compute_decayed_importance(self.importance, elapsed, config)

    def to_str(self) -> str:
        """
        Creates a concise, formatted string representation of the MemoryItem.
        """
        sections = [
            f"### Memory: {self.memory_id}",
            self._format_content(),
            self._format_summary(),
            self._format_details(),
            self._format_metrics(),
            self._format_emotional(),
            self._format_connections(),
            self._format_timeline(),
            self._format_source(),
        ]
        sections.extend(self._format_extra())
        return "\n".join(s for s in sections if s)

    def _format_content(self) -> str | None:
        return f"```\n{self.content}\n```" if self.content else None

    def _format_summary(self) -> str | None:
        return f"**Summary:** {self.summary}" if self.summary else None

    @staticmethod
    def _format_time(value: datetime | str | None) -> str:
        if value is None:
            return "N/A"
        dt = coerce_datetime(value)
        if dt is None:
            return str(value)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _format_float(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "N/A"

    @staticmethod
    def _format_rating(value: float | None) -> str:
        if value is None:
            return ""
        filled = round(value * 5)
        return f"[{'|' * filled}{'-' * (5 - filled)}]"

    def _format_details(self) -> str:
        details = [
            f"**Type:** {self.memory_type}",
            f"**Created:** {self._format_time(self.timestamp)}",
        ]
        if self.tags:
            details.append(f"**Tags:** {', '.join(self.tags)}")
        return " | ".join(details)

    def _format_metrics(self) -> str | None:
        metrics = []
        if self.importance is not None:
            metrics.append(
                f"**Imp:** {self._format_float(self.importance)} {self._format_rating(self.importance)}"
            )
        if self.confidence is not None:
            metrics.append(
                f"**Conf:** {self._format_float(self.confidence)} {self._format_rating(self.confidence)}"
            )
        if self.fidelity is not None:
            metrics.append(
                f"**Fid:** {self._format_float(self.fidelity)} {self._format_rating(self.fidelity)}"
            )
        return " | ".join(metrics) if metrics else None

    def _format_emotional(self) -> str | None:
        if self.emotional_valence is None and self.emotional_arousal is None:
            return None
        emotional = []
        if self.emotional_valence is not None:
            if self.emotional_valence > 0:
                sentiment = "(+)"
            elif self.emotional_valence < 0:
                sentiment = "(-)"
            else:
                sentiment = "(=)"
            emotional.append(f"**Val:** {self._format_float(self.emotional_valence)}{sentiment}")
        if self.emotional_arousal is not None:
            emotional.append(f"**Aro:** {self._format_float(self.emotional_arousal)}")
        if self.emotions:
            emotional.append(f"**Emo:** {', '.join(self.emotions)}")
        return " | ".join(emotional) if emotional else None

    def _format_connections(self) -> str | None:
        if not (self.associations or self.linked_concepts or self.conflicts_with):
            return None
        connections = []
        if self.associations:
            connections.append(f"**Assoc:** {', '.join(self.associations)}")
        if self.linked_concepts:
            connections.append(f"**Links:** {', '.join(self.linked_concepts)}")
        if self.conflicts_with:
            connections.append(f"**Conflicts:** {', '.join(self.conflicts_with)}")
        return " | ".join(connections)

    def _format_timeline(self) -> str | None:
        if not (self.previous_event_id or self.next_event_id):
            return None
        timeline = []
        if self.previous_event_id:
            timeline.append(f"← {self.previous_event_id}")
        if self.next_event_id:
            timeline.append(f"{self.next_event_id} →")
        return "**Timeline:** " + " | ".join(timeline)

    def _format_source(self) -> str | None:
        if not (self.source or self.credibility is not None):
            return None
        source_info = []
        if self.source:
            source_info.append(f"**Source:** {self.source}")
        if self.credibility is not None:
            source_info.append(
                f"**Cred:** {self._format_float(self.credibility)} {self._format_rating(self.credibility)}"
            )
        return " | ".join(source_info)

    def _format_extra(self) -> list[str]:
        extra: list[str] = []
        if self.context:
            extra.append(f"**Context:** ```{json.dumps(self.context)}```")
        if self.metadata:
            extra.append(f"**Metadata:** ```{json.dumps(self.metadata)}```")
        return extra

    def _format_emotional_context_llm(self) -> str | None:
        """Format emotional context for LLM consumption."""
        has_emotional_data = (
            self.emotional_valence is not None
            or self.emotional_arousal is not None
            or self.emotions
        )
        if not has_emotional_data:
            return None

        emotion_parts = []
        if self.emotional_valence is not None:
            emotion_parts.append(f"valence: {self.emotional_valence:+.2f}")
        if self.emotional_arousal is not None:
            emotion_parts.append(f"arousal: {self.emotional_arousal:.2f}")
        if self.emotions:
            emotion_parts.append(f"emotions: {', '.join(self.emotions)}")
        return f"Emotional Context: {' | '.join(emotion_parts)}"

    def _format_timeline_llm(self) -> str | None:
        """Format temporal context for LLM consumption."""
        if not (self.previous_event_id or self.next_event_id):
            return None

        timeline = []
        if self.previous_event_id:
            timeline.append(f"previous: {self.previous_event_id}")
        if self.next_event_id:
            timeline.append(f"next: {self.next_event_id}")
        return f"Timeline: {' | '.join(timeline)}"

    def to_str_llm(self) -> str:
        """
        Creates a simplified string representation optimized for LLM consumption.
        Focuses on core information and critical context while maintaining a clean format.
        """
        parts = [
            f"Memory ({self.memory_type.value}):",
            f"Content: {self.content}",
        ]

        if self.summary:
            parts.append(f"Summary: {self.summary}")

        parts.append(f"Importance: {self.importance:.2f}")

        emotional_context = self._format_emotional_context_llm()
        if emotional_context:
            parts.append(emotional_context)

        if self.linked_concepts:
            parts.append(f"Related Concepts: {', '.join(self.linked_concepts)}")

        if self.entities:
            parts.append(f"Entities: {', '.join([f'{k} ({v})' for k, v in self.entities.items()])}")

        timeline = self._format_timeline_llm()
        if timeline:
            parts.append(timeline)

        return "\n".join(parts)

    def to_langchain_document(self) -> Any:
        """
        Convert the MemoryItem to a LangChain Document.
        Requires langchain to be installed.

        Returns:
            Document: A LangChain Document containing the memory content and metadata
        """
        try:
            from langchain.schema import Document
        except ImportError as err:
            raise ImportError(
                "langchain package is required to use this method. "
                "Please install it with `pip install langchain`"
            ) from err

        # Prepare metadata dictionary with all relevant fields
        metadata = {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "memory_category": self.memory_type.category,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "confidence": self.confidence,
            "fidelity": self.fidelity,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
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
            "repo_id": self.repo_id,
            "worktree_id": self.worktree_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }

        # Add any custom context and metadata
        metadata.update(self.metadata)

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(page_content=self.content, metadata=metadata)


class MemoryLink(BaseModel):
    """A directed link between memories with context.

    Represents a semantic relationship between two memories, inspired by Zettelkasten methodology.
    Links have types, strength (which can decay), context explaining why the link exists, and
    metadata about how the link was created.

    Attributes:
        link_id (str): Unique identifier for the link
        source_id (str): ID of the source memory
        target_id (str): ID of the target memory
        link_type (LinkType): Semantic type of the relationship
        strength (float): Link strength (0-1), can decay over time via FSRS
        context (Optional[str]): Explanation of why this link exists
        created_at (datetime): When the link was created
        last_accessed (datetime): Last time the link was traversed
        access_count (int): Number of times the link was accessed
        created_by (Literal): How the link was created (user, auto_similarity, auto_entity, llm)
        similarity_score (Optional[float]): Similarity score if created automatically
        metadata (Dict[str, Any]): Additional flexible attributes
    """

    link_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    link_type: LinkType
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    context: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    created_by: Literal["user", "auto_similarity", "auto_entity", "llm"] = "user"
    similarity_score: float | None = Field(None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def update_access(self):
        """Update access tracking when link is traversed."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        # Strengthen link slightly on access (similar to memory reinforcement)
        self.strength = min(1.0, self.strength + 0.05)

    def decay_strength(self, current_time: datetime, decay_rate: float = 0.01):
        """Apply time-based decay to link strength.

        Args:
            current_time: Current datetime for calculating elapsed time
            decay_rate: Rate of decay (default 0.01)
        """
        elapsed_seconds = max(0.0, (current_time - self.last_accessed).total_seconds())
        decay_factor = 1.0 / (1.0 + decay_rate * elapsed_seconds)
        self.strength *= decay_factor
