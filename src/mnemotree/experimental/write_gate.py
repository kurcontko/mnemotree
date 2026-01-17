"""
Context-aware write gating policy for memory storage.

Implements intelligent filtering to avoid storing noise:
1. Novelty threshold - avoid redundant information
2. User preferences - respect storage policies
3. Confidence thresholds - filter low-quality memories
4. Content quality assessment - semantic meaningfulness
5. Relevance scoring - contextual importance
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ..core.models import MemoryItem, MemoryType


class WriteDecision(str, Enum):
    """Decision outcomes for write gating."""

    ACCEPT = "accept"  # Store the memory
    REJECT = "reject"  # Don't store
    MERGE = "merge"  # Merge with existing
    DEFER = "defer"  # Needs more context
    REQUEST_APPROVAL = "request_approval"  # Ask user


class RejectionReason(str, Enum):
    """Reasons for rejecting a memory write."""

    TOO_SIMILAR = "too_similar"  # Redundant
    LOW_QUALITY = "low_quality"  # Poor content
    LOW_CONFIDENCE = "low_confidence"  # Uncertain
    IRRELEVANT = "irrelevant"  # Not contextually relevant
    PRIVACY_CONCERN = "privacy_concern"  # Sensitive data
    USER_PREFERENCE = "user_preference"  # User settings
    TOO_SHORT = "too_short"  # Insufficient content
    NOISE = "noise"  # Likely noise/spam


@dataclass
class WritePolicy:
    """
    Policy configuration for write gating.

    Defines thresholds and rules for accepting/rejecting memories.
    """

    # Novelty requirements
    min_novelty_score: float = 0.3  # 0 = duplicate, 1 = completely novel
    allow_redundant: bool = False

    # Quality requirements
    min_confidence: float = 0.5
    min_content_length: int = 10
    min_importance: float = 0.1

    # Content filtering
    require_meaningful_content: bool = True
    block_pii: bool = False  # Block personally identifiable information

    # User preferences
    memory_types_allowed: list[MemoryType] | None = None
    max_memories_per_hour: int | None = None

    # Override rules
    always_store_high_importance: float = 0.8  # Always store above this
    auto_approve_semantic: bool = True  # Auto-approve semantic memories

    @classmethod
    def permissive(cls) -> "WritePolicy":
        """Create a permissive policy (store most things)."""
        return cls(
            min_novelty_score=0.1,
            min_confidence=0.3,
            min_content_length=5,
            min_importance=0.0,
            require_meaningful_content=False,
        )

    @classmethod
    def strict(cls) -> "WritePolicy":
        """Create a strict policy (store only high-quality)."""
        return cls(
            min_novelty_score=0.6,
            min_confidence=0.7,
            min_content_length=20,
            min_importance=0.3,
            require_meaningful_content=True,
            allow_redundant=False,
        )

    @classmethod
    def balanced(cls) -> "WritePolicy":
        """Create a balanced policy (default)."""
        return cls()


@dataclass
class WriteResult:
    """Result of write gating evaluation."""

    decision: WriteDecision
    reasons: list[str]
    score: float  # Overall acceptance score

    # Details
    novelty_score: float | None = None
    quality_score: float | None = None
    relevance_score: float | None = None

    # Recommendations
    suggested_merge_id: str | None = None
    modifications: dict[str, Any] | None = None


class ContextAwareWriteGate:
    """
    Intelligent write gating system for memory storage.

    Evaluates whether a memory should be stored based on:
    - Novelty (is this new information?)
    - Quality (is this well-formed and meaningful?)
    - Relevance (does this matter in current context?)
    - User preferences (does user want this?)
    - Confidence (how certain are we?)
    """

    def __init__(
        self,
        policy: WritePolicy | None = None,
        novelty_assessor: Callable | None = None,
    ):
        """
        Initialize write gate.

        Args:
            policy: Write policy configuration
            novelty_assessor: Optional custom novelty assessment function
        """
        self.policy = policy or WritePolicy.balanced()
        self.novelty_assessor = novelty_assessor

        # Rate limiting tracking
        self._write_timestamps: dict[str, list[datetime]] = {}

    async def evaluate(
        self,
        memory: MemoryItem,
        context: dict[str, Any] | None = None,
        existing_memories: list[MemoryItem] | None = None,
    ) -> WriteResult:
        """
        Evaluate whether to store a memory.

        Args:
            memory: Memory to evaluate
            context: Optional context for evaluation
            existing_memories: Optional list of existing memories for comparison

        Returns:
            WriteResult with decision and reasoning
        """
        reasons: list[str] = []
        scores: dict[str, float] = {}

        override = self._override_decision(memory)
        if override:
            return override

        policy_gate = self._policy_memory_type_gate(memory)
        if policy_gate:
            return policy_gate

        rate_gate = self._rate_limit_gate(memory)
        if rate_gate:
            return rate_gate

        quality_gate = self._apply_quality_gate(memory, scores, reasons)
        if quality_gate:
            return quality_gate

        novelty_gate = await self._apply_novelty_gate(memory, existing_memories, scores, reasons)
        if novelty_gate:
            return novelty_gate

        meaningful_gate = self._apply_meaningfulness_gate(memory, scores, reasons)
        if meaningful_gate:
            return meaningful_gate

        pii_gate = self._apply_pii_gate(memory)
        if pii_gate:
            return pii_gate

        if context:
            self._apply_relevance(memory, context, scores, reasons)

        return self._finalize_decision(scores, reasons)

    def _override_decision(self, memory: MemoryItem) -> WriteResult | None:
        if memory.importance >= self.policy.always_store_high_importance:
            return WriteResult(
                decision=WriteDecision.ACCEPT,
                reasons=["High importance override"],
                score=1.0,
            )
        if self.policy.auto_approve_semantic and memory.memory_type == MemoryType.SEMANTIC:
            return WriteResult(
                decision=WriteDecision.ACCEPT,
                reasons=["Semantic memory auto-approval"],
                score=0.9,
            )
        return None

    def _policy_memory_type_gate(self, memory: MemoryItem) -> WriteResult | None:
        if (
            self.policy.memory_types_allowed
            and memory.memory_type not in self.policy.memory_types_allowed
        ):
            return WriteResult(
                decision=WriteDecision.REJECT,
                reasons=[f"Memory type {memory.memory_type} not allowed by policy"],
                score=0.0,
            )
        return None

    def _rate_limit_gate(self, memory: MemoryItem) -> WriteResult | None:
        if self.policy.max_memories_per_hour and not self._check_rate_limit(
            memory.user_id or "default"
        ):
            return WriteResult(
                decision=WriteDecision.DEFER,
                reasons=["Rate limit exceeded"],
                score=0.0,
            )
        return None

    def _apply_quality_gate(
        self,
        memory: MemoryItem,
        scores: dict[str, float],
        reasons: list[str],
    ) -> WriteResult | None:
        quality_result = self._check_quality(memory)
        if quality_result.decision == WriteDecision.REJECT:
            return quality_result
        scores["quality"] = quality_result.quality_score or 0.5
        reasons.extend(quality_result.reasons)
        return None

    async def _apply_novelty_gate(
        self,
        memory: MemoryItem,
        existing_memories: list[MemoryItem] | None,
        scores: dict[str, float],
        reasons: list[str],
    ) -> WriteResult | None:
        novelty_result = await self._assess_novelty(memory, existing_memories)
        scores["novelty"] = novelty_result.novelty_score or 0.5
        if novelty_result.decision in (WriteDecision.REJECT, WriteDecision.MERGE):
            return novelty_result
        reasons.extend(novelty_result.reasons)
        return None

    def _apply_meaningfulness_gate(
        self,
        memory: MemoryItem,
        scores: dict[str, float],
        reasons: list[str],
    ) -> WriteResult | None:
        if not self.policy.require_meaningful_content:
            return None
        meaningful_result = self._check_meaningful_content(memory)
        if meaningful_result.decision == WriteDecision.REJECT:
            return meaningful_result
        scores["meaningfulness"] = meaningful_result.score
        reasons.extend(meaningful_result.reasons)
        return None

    def _apply_pii_gate(self, memory: MemoryItem) -> WriteResult | None:
        if not self.policy.block_pii:
            return None
        pii_result = self._check_pii(memory)
        if pii_result.decision == WriteDecision.REJECT:
            return pii_result
        return None

    def _apply_relevance(
        self,
        memory: MemoryItem,
        context: dict[str, Any],
        scores: dict[str, float],
        reasons: list[str],
    ) -> None:
        relevance_result = self._assess_relevance(memory, context)
        scores["relevance"] = relevance_result.relevance_score or 0.5
        reasons.extend(relevance_result.reasons)

    def _finalize_decision(
        self,
        scores: dict[str, float],
        reasons: list[str],
    ) -> WriteResult:
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5
        if overall_score >= 0.6:
            decision = WriteDecision.ACCEPT
        elif overall_score >= 0.4:
            decision = WriteDecision.REQUEST_APPROVAL
        else:
            decision = WriteDecision.REJECT
        return WriteResult(
            decision=decision,
            reasons=reasons,
            score=overall_score,
            novelty_score=scores.get("novelty"),
            quality_score=scores.get("quality"),
            relevance_score=scores.get("relevance"),
        )

    def _check_quality(self, memory: MemoryItem) -> WriteResult:
        """Check basic quality criteria."""
        reasons = []

        # Length check
        if len(memory.content.strip()) < self.policy.min_content_length:
            return WriteResult(
                decision=WriteDecision.REJECT,
                reasons=[RejectionReason.TOO_SHORT.value],
                score=0.0,
            )

        # Confidence check
        if memory.confidence < self.policy.min_confidence:
            return WriteResult(
                decision=WriteDecision.REJECT,
                reasons=[RejectionReason.LOW_CONFIDENCE.value],
                score=0.0,
            )

        # Importance check
        if memory.importance < self.policy.min_importance:
            reasons.append("Low importance")

        # Calculate quality score
        quality_score = (
            0.4 * memory.confidence
            + 0.3 * memory.importance
            + 0.3 * min(1.0, len(memory.content) / 100.0)
        )

        return WriteResult(
            decision=WriteDecision.ACCEPT,
            reasons=reasons,
            score=quality_score,
            quality_score=quality_score,
        )

    async def _assess_novelty(
        self, memory: MemoryItem, existing_memories: list[MemoryItem] | None = None
    ) -> WriteResult:
        """Assess novelty of memory content."""
        if not existing_memories:
            # No comparison data - assume novel
            return WriteResult(
                decision=WriteDecision.ACCEPT,
                reasons=["No existing memories to compare"],
                score=1.0,
                novelty_score=1.0,
            )

        # Use custom novelty assessor if provided
        if self.novelty_assessor:
            novelty_score = await self.novelty_assessor(memory, existing_memories)
        else:
            novelty_score = self._default_novelty_assessment(memory, existing_memories)

        # Check against policy threshold
        if novelty_score < self.policy.min_novelty_score and not self.policy.allow_redundant:
            # Find most similar memory for potential merge
            similar_memory = self._find_most_similar(memory, existing_memories)

            return WriteResult(
                decision=WriteDecision.MERGE,
                reasons=[RejectionReason.TOO_SIMILAR.value],
                score=novelty_score,
                novelty_score=novelty_score,
                suggested_merge_id=similar_memory.memory_id if similar_memory else None,
            )

        return WriteResult(
            decision=WriteDecision.ACCEPT,
            reasons=[f"Novelty score: {novelty_score:.2f}"],
            score=novelty_score,
            novelty_score=novelty_score,
        )

    def _default_novelty_assessment(
        self, memory: MemoryItem, existing_memories: list[MemoryItem]
    ) -> float:
        """Default novelty assessment using embeddings."""
        if not memory.embedding:
            return 0.5  # Unknown

        # Compare to recent memories (last 50)
        recent_memories = sorted(existing_memories, key=lambda m: m.timestamp, reverse=True)[:50]

        max_similarity = 0.0
        for existing in recent_memories:
            if not existing.embedding:
                continue

            similarity = self._cosine_similarity(memory.embedding, existing.embedding)
            max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        novelty_score = 1.0 - max_similarity
        return novelty_score

    def _find_most_similar(
        self, memory: MemoryItem, existing_memories: list[MemoryItem]
    ) -> MemoryItem | None:
        """Find most similar existing memory."""
        if not memory.embedding:
            return None

        max_sim = 0.0
        most_similar = None

        for existing in existing_memories:
            if not existing.embedding:
                continue

            sim = self._cosine_similarity(memory.embedding, existing.embedding)
            if sim > max_sim:
                max_sim = sim
                most_similar = existing

        return most_similar

    def _check_meaningful_content(self, memory: MemoryItem) -> WriteResult:
        """Check if content is meaningful (not noise/spam)."""
        content = memory.content.strip().lower()

        # Heuristic checks for meaningless content
        if len(set(content.split())) < 3:
            # Too few unique words
            return WriteResult(
                decision=WriteDecision.REJECT,
                reasons=[RejectionReason.NOISE.value],
                score=0.0,
            )

        # Check for excessive punctuation/symbols
        symbol_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content)
        if symbol_ratio > 0.3:
            return WriteResult(
                decision=WriteDecision.REJECT,
                reasons=[RejectionReason.NOISE.value],
                score=0.0,
            )

        # Check for actual sentences (very basic)
        has_verb = any(
            word in content for word in ["is", "are", "was", "were", "have", "has", "do", "does"]
        )
        if not has_verb and len(content) > 20:
            return WriteResult(
                decision=WriteDecision.REJECT,
                reasons=[RejectionReason.LOW_QUALITY.value],
                score=0.3,
            )

        return WriteResult(
            decision=WriteDecision.ACCEPT,
            reasons=["Content appears meaningful"],
            score=0.8,
        )

    def _check_pii(self, memory: MemoryItem) -> WriteResult:
        """Check for personally identifiable information."""
        import re

        content = memory.content

        # Simple regex patterns for common PII
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

        for pii_type, pattern in patterns.items():
            if re.search(pattern, content):
                return WriteResult(
                    decision=WriteDecision.REJECT,
                    reasons=[f"{RejectionReason.PRIVACY_CONCERN.value}: {pii_type} detected"],
                    score=0.0,
                )

        return WriteResult(
            decision=WriteDecision.ACCEPT,
            reasons=[],
            score=1.0,
        )

    def _assess_relevance(self, memory: MemoryItem, context: dict[str, Any]) -> WriteResult:
        """Assess contextual relevance."""
        relevance_score = 0.5  # Default neutral
        reasons = []

        # Check context hints
        if "user_query" in context:
            # Memory relates to active user query
            relevance_score += 0.2
            reasons.append("Relates to active query")

        if "conversation_id" in context:
            # Part of ongoing conversation
            relevance_score += 0.1
            reasons.append("Part of conversation")

        if "important_topics" in context:
            # Check if memory mentions important topics
            topics = context["important_topics"]
            if any(topic.lower() in memory.content.lower() for topic in topics):
                relevance_score += 0.2
                reasons.append("Mentions important topics")

        relevance_score = min(1.0, relevance_score)

        return WriteResult(
            decision=WriteDecision.ACCEPT,
            reasons=reasons,
            score=relevance_score,
            relevance_score=relevance_score,
        )

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if rate limit is exceeded."""
        if user_id not in self._write_timestamps:
            self._write_timestamps[user_id] = []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        # Clean old timestamps
        self._write_timestamps[user_id] = [
            ts for ts in self._write_timestamps[user_id] if ts > cutoff
        ]

        # Check limit
        if len(self._write_timestamps[user_id]) >= self.policy.max_memories_per_hour:
            return False

        # Record this write
        self._write_timestamps[user_id].append(now)
        return True

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))
