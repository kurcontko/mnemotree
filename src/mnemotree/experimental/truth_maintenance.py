"""
Conflict detection and truth maintenance system.

Implements:
1. Claims registry - normalized facts with timestamps and confidence
2. Contradiction detection between claims
3. Staleness tracking and temporal validity
4. Conflict resolution strategies
5. Truth maintenance and claim lifecycle management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, ConfigDict, Field

from ..core.models import MemoryItem


class ClaimStatus(str, Enum):
    """Status of a claim in the registry."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    CONFLICTED = "conflicted"
    SUPERSEDED = "superseded"
    UNVERIFIED = "unverified"


class ConflictSeverity(str, Enum):
    """Severity level of conflicts."""

    LOW = "low"  # Minor inconsistency
    MEDIUM = "medium"  # Significant contradiction
    HIGH = "high"  # Direct contradiction
    CRITICAL = "critical"  # Critical factual error


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""

    MOST_RECENT = "most_recent"  # Prefer newest claim
    HIGHEST_CONFIDENCE = "highest_confidence"  # Prefer most confident
    MOST_ACCESSED = "most_accessed"  # Prefer most used
    MANUAL = "manual"  # Require human intervention
    ENSEMBLE = "ensemble"  # Combine multiple signals


@dataclass
class Claim(BaseModel):
    """
    A normalized fact or statement extracted from memories.

    Claims are atomic, verifiable statements that can be:
    - Tracked for contradictions
    - Updated with new information
    - Superseded by better information
    - Retired when no longer relevant
    """

    claim_id: str = Field(default_factory=lambda: str(uuid4()))

    # Core content
    subject: str  # What the claim is about
    predicate: str  # The relationship or property
    object: str  # The value or target

    # Full statement
    statement: str  # Natural language form

    # Provenance
    source_memory_ids: list[str] = Field(default_factory=list)

    # Temporal validity
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Quality metrics
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    credibility: float = Field(default=1.0, ge=0.0, le=1.0)

    # Status
    status: ClaimStatus = ClaimStatus.ACTIVE
    superseded_by: str | None = None  # ID of claim that supersedes this

    # Metadata
    context: dict[str, Any] = Field(default_factory=dict)
    access_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class Conflict:
    """Detected conflict between claims."""

    conflict_id: str = field(default_factory=lambda: str(uuid4()))

    # Conflicting claims
    claim_ids: list[str] = field(default_factory=list)

    # Conflict details
    conflict_type: str = "contradiction"  # "contradiction", "inconsistency", "ambiguity"
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    description: str = ""

    # Resolution
    resolved: bool = False
    resolution_strategy: ResolutionStrategy | None = None
    resolved_at: datetime | None = None
    winner_claim_id: str | None = None

    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class ClaimsRegistry:
    """
    Registry for managing claims, detecting conflicts, and maintaining truth.

    This is the core of the truth maintenance system, tracking:
    - Individual claims extracted from memories
    - Relationships and conflicts between claims
    - Temporal validity and staleness
    - Confidence and credibility evolution
    """

    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
        staleness_threshold_days: int = 90,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize claims registry.

        Args:
            llm: Language model for claim extraction and conflict detection
            staleness_threshold_days: Days after which claims may be stale
            confidence_threshold: Minimum confidence for active claims
        """
        self.llm = llm
        self.staleness_threshold_days = staleness_threshold_days
        self.confidence_threshold = confidence_threshold

        # Storage
        self.claims: dict[str, Claim] = {}
        self.conflicts: dict[str, Conflict] = {}

        # Indices for efficient querying
        self._subject_index: dict[str, set[str]] = {}  # subject -> claim_ids
        self._memory_index: dict[str, set[str]] = {}  # memory_id -> claim_ids

    async def extract_claims(self, memory: MemoryItem) -> list[Claim]:
        """
        Extract atomic claims from a memory.

        Args:
            memory: Memory to extract claims from

        Returns:
            List of extracted claims
        """
        if not self.llm:
            # Fallback: create a single claim from the memory
            return [
                Claim(
                    subject="memory",
                    predicate="contains",
                    object=memory.content[:100],
                    statement=memory.content,
                    source_memory_ids=[memory.memory_id],
                    confidence=memory.confidence,
                    credibility=memory.credibility or 1.0,
                )
            ]

        # Use LLM to extract structured claims
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            """Extract atomic, verifiable claims from the following text.
For each claim, provide:
- Subject (what the claim is about)
- Predicate (the relationship or property)
- Object (the value or target)
- Statement (natural language form)

Text: {content}

Format each claim as: SUBJECT | PREDICATE | OBJECT | STATEMENT

Claims:"""
        )

        chain = prompt | self.llm
        result = await chain.ainvoke({"content": memory.content})
        text = result.content if hasattr(result, "content") else str(result)

        # Parse claims from LLM output
        claims = []
        for line in text.strip().split("\n"):
            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                claims.append(
                    Claim(
                        subject=parts[0],
                        predicate=parts[1],
                        object=parts[2],
                        statement=parts[3],
                        source_memory_ids=[memory.memory_id],
                        confidence=memory.confidence,
                        credibility=memory.credibility or 1.0,
                    )
                )

        return (
            claims
            if claims
            else [
                Claim(
                    subject="memory",
                    predicate="contains",
                    object=memory.content[:100],
                    statement=memory.content,
                    source_memory_ids=[memory.memory_id],
                    confidence=memory.confidence,
                )
            ]
        )

    async def register_claims(self, claims: list[Claim]) -> list[str]:
        """
        Register new claims and check for conflicts.

        Args:
            claims: Claims to register

        Returns:
            List of registered claim IDs
        """
        registered_ids = []

        for claim in claims:
            # Store claim
            self.claims[claim.claim_id] = claim
            registered_ids.append(claim.claim_id)

            # Update indices
            self._update_indices(claim)

            # Check for conflicts with existing claims
            conflicts = await self._detect_conflicts(claim)
            for conflict in conflicts:
                self.conflicts[conflict.conflict_id] = conflict

        return registered_ids

    async def _detect_conflicts(self, new_claim: Claim) -> list[Conflict]:
        """
        Detect conflicts between a new claim and existing claims.

        Args:
            new_claim: Newly registered claim

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Get potentially conflicting claims (same subject)
        related_claim_ids = self._subject_index.get(new_claim.subject, set())

        for claim_id in related_claim_ids:
            if claim_id == new_claim.claim_id:
                continue

            existing_claim = self.claims.get(claim_id)
            if not existing_claim or existing_claim.status != ClaimStatus.ACTIVE:
                continue

            # Check for conflicts
            conflict = await self._check_conflict(new_claim, existing_claim)
            if conflict:
                conflicts.append(conflict)

        return conflicts

    async def _check_conflict(self, claim1: Claim, claim2: Claim) -> Conflict | None:
        """
        Check if two claims conflict.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Conflict object if conflicting, None otherwise
        """
        # Same subject and predicate but different objects = potential conflict
        if (
            claim1.subject.lower() == claim2.subject.lower()
            and claim1.predicate.lower() == claim2.predicate.lower()
            and claim1.object.lower() != claim2.object.lower()
        ):
            # Use LLM for nuanced conflict detection
            if self.llm:
                severity = await self._assess_conflict_severity(claim1, claim2)
            else:
                severity = ConflictSeverity.MEDIUM

            return Conflict(
                claim_ids=[claim1.claim_id, claim2.claim_id],
                conflict_type="contradiction",
                severity=severity,
                description=f"{claim1.subject} has conflicting {claim1.predicate}: '{claim1.object}' vs '{claim2.object}'",
            )

        return None

    async def _assess_conflict_severity(self, claim1: Claim, claim2: Claim) -> ConflictSeverity:
        """Use LLM to assess conflict severity."""
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            """Assess the severity of the conflict between these two claims:

Claim 1: {statement1}
Claim 2: {statement2}

Are these claims:
- CRITICAL: Directly contradictory facts
- HIGH: Strong contradiction
- MEDIUM: Moderate inconsistency
- LOW: Minor difference or contextual variation

Severity:"""
        )

        chain = prompt | self.llm
        result = await chain.ainvoke(
            {
                "statement1": claim1.statement,
                "statement2": claim2.statement,
            }
        )

        if hasattr(result, "content"):
            text = result.content.strip().upper()
        else:
            text = str(result).strip().upper()

        if "CRITICAL" in text:
            return ConflictSeverity.CRITICAL
        elif "HIGH" in text:
            return ConflictSeverity.HIGH
        elif "LOW" in text:
            return ConflictSeverity.LOW
        else:
            return ConflictSeverity.MEDIUM

    async def resolve_conflict(
        self, conflict_id: str, strategy: ResolutionStrategy = ResolutionStrategy.MOST_RECENT
    ) -> str | None:
        """
        Resolve a conflict using the specified strategy.

        Args:
            conflict_id: Conflict to resolve
            strategy: Resolution strategy to use

        Returns:
            ID of winning claim, or None if unresolved
        """
        conflict = self.conflicts.get(conflict_id)
        if not conflict or conflict.resolved:
            return None

        claims = [self.claims[cid] for cid in conflict.claim_ids if cid in self.claims]
        if not claims:
            return None

        # Apply resolution strategy
        if strategy == ResolutionStrategy.MOST_RECENT:
            winner = max(claims, key=lambda c: c.updated_at)
        elif strategy == ResolutionStrategy.HIGHEST_CONFIDENCE:
            winner = max(claims, key=lambda c: c.confidence)
        elif strategy == ResolutionStrategy.MOST_ACCESSED:
            winner = max(claims, key=lambda c: c.access_count)
        elif strategy == ResolutionStrategy.ENSEMBLE:
            # Weighted combination
            scores = []
            for claim in claims:
                recency_score = (datetime.now(timezone.utc) - claim.updated_at).days
                recency_score = 1.0 / (1.0 + recency_score / 30.0)

                score = (
                    0.4 * claim.confidence
                    + 0.3 * recency_score
                    + 0.3 * min(1.0, claim.access_count / 10.0)
                )
                scores.append(score)
            winner = claims[scores.index(max(scores))]
        else:
            return None  # Manual resolution required

        # Update claim statuses
        for claim in claims:
            if claim.claim_id != winner.claim_id:
                claim.status = ClaimStatus.SUPERSEDED
                claim.superseded_by = winner.claim_id

        # Mark conflict as resolved
        conflict.resolved = True
        conflict.resolution_strategy = strategy
        conflict.resolved_at = datetime.now(timezone.utc)
        conflict.winner_claim_id = winner.claim_id

        return winner.claim_id

    def check_staleness(self) -> list[str]:
        """
        Check for stale claims that may need updating.

        Returns:
            List of claim IDs that are potentially stale
        """
        stale_ids = []
        threshold = datetime.now(timezone.utc) - timedelta(days=self.staleness_threshold_days)

        for claim_id, claim in self.claims.items():
            if claim.status != ClaimStatus.ACTIVE:
                continue

            # Check if claim is old and hasn't been accessed recently
            if claim.updated_at < threshold and claim.access_count < 5:
                stale_ids.append(claim_id)

        return stale_ids

    def get_active_conflicts(self) -> list[Conflict]:
        """Get all unresolved conflicts."""
        return [conflict for conflict in self.conflicts.values() if not conflict.resolved]

    def get_claims_by_subject(self, subject: str) -> list[Claim]:
        """Get all active claims about a subject."""
        claim_ids = self._subject_index.get(subject, set())
        return [
            self.claims[cid]
            for cid in claim_ids
            if cid in self.claims and self.claims[cid].status == ClaimStatus.ACTIVE
        ]

    def get_claims_for_memory(self, memory_id: str) -> list[Claim]:
        """Get all claims derived from a memory."""
        claim_ids = self._memory_index.get(memory_id, set())
        return [self.claims[cid] for cid in claim_ids if cid in self.claims]

    def _update_indices(self, claim: Claim):
        """Update internal indices for a claim."""
        # Subject index
        if claim.subject not in self._subject_index:
            self._subject_index[claim.subject] = set()
        self._subject_index[claim.subject].add(claim.claim_id)

        # Memory index
        for memory_id in claim.source_memory_ids:
            if memory_id not in self._memory_index:
                self._memory_index[memory_id] = set()
            self._memory_index[memory_id].add(claim.claim_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics."""
        total_claims = len(self.claims)
        active_claims = sum(1 for c in self.claims.values() if c.status == ClaimStatus.ACTIVE)
        total_conflicts = len(self.conflicts)
        unresolved_conflicts = sum(1 for c in self.conflicts.values() if not c.resolved)

        return {
            "total_claims": total_claims,
            "active_claims": active_claims,
            "deprecated_claims": total_claims - active_claims,
            "total_conflicts": total_conflicts,
            "unresolved_conflicts": unresolved_conflicts,
            "resolved_conflicts": total_conflicts - unresolved_conflicts,
        }
