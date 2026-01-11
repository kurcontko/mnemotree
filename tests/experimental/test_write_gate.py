"""Tests for ContextAwareWriteGate and WritePolicy."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.experimental.write_gate import (
    ContextAwareWriteGate,
    RejectionReason,
    WriteDecision,
    WritePolicy,
    WriteResult,
)


def _make_memory(
    content: str = "This is a test memory with some meaningful content.",
    importance: float = 0.5,
    confidence: float = 0.8,
    memory_type: MemoryType = MemoryType.EPISODIC,
    embedding: list[float] | None = None,
) -> MemoryItem:
    """Create a test memory item."""
    return MemoryItem(
        memory_id="test-123",
        content=content,
        memory_type=memory_type,
        importance=importance,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        embedding=embedding or [0.1] * 10,
    )


class TestWritePolicy:
    """Tests for WritePolicy dataclass."""

    def test_default_values(self):
        """WritePolicy has sensible defaults."""
        policy = WritePolicy()
        assert abs(policy.min_novelty_score - 0.3) < 1e-9
        assert abs(policy.min_confidence - 0.5) < 1e-9
        assert policy.min_content_length == 10
        assert policy.require_meaningful_content is True

    def test_permissive_policy(self):
        """WritePolicy.permissive() creates low thresholds."""
        policy = WritePolicy.permissive()
        assert abs(policy.min_novelty_score - 0.1) < 1e-9
        assert abs(policy.min_confidence - 0.3) < 1e-9
        assert policy.require_meaningful_content is False

    def test_strict_policy(self):
        """WritePolicy.strict() creates high thresholds."""
        policy = WritePolicy.strict()
        assert abs(policy.min_novelty_score - 0.6) < 1e-9
        assert abs(policy.min_confidence - 0.7) < 1e-9
        assert policy.min_content_length == 20

    def test_balanced_policy(self):
        """WritePolicy.balanced() returns default policy."""
        policy = WritePolicy.balanced()
        assert abs(policy.min_novelty_score - 0.3) < 1e-9  # Same as default


class TestWriteDecision:
    """Tests for WriteDecision enum."""

    def test_all_decisions_exist(self):
        """All expected decision types are defined."""
        assert WriteDecision.ACCEPT.value == "accept"
        assert WriteDecision.REJECT.value == "reject"
        assert WriteDecision.MERGE.value == "merge"
        assert WriteDecision.DEFER.value == "defer"
        assert WriteDecision.REQUEST_APPROVAL.value == "request_approval"


class TestRejectionReason:
    """Tests for RejectionReason enum."""

    def test_rejection_reasons_exist(self):
        """Common rejection reasons are defined."""
        assert RejectionReason.TOO_SIMILAR.value == "too_similar"
        assert RejectionReason.LOW_QUALITY.value == "low_quality"
        assert RejectionReason.LOW_CONFIDENCE.value == "low_confidence"
        assert RejectionReason.TOO_SHORT.value == "too_short"
        assert RejectionReason.PRIVACY_CONCERN.value == "privacy_concern"


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_creation(self):
        """WriteResult can be created with required fields."""
        result = WriteResult(
            decision=WriteDecision.ACCEPT,
            reasons=["High importance"],
            score=0.9,
        )
        assert result.decision == WriteDecision.ACCEPT
        assert abs(result.score - 0.9) < 1e-9

    def test_optional_fields(self):
        """WriteResult has optional score fields."""
        result = WriteResult(
            decision=WriteDecision.REJECT,
            reasons=["Too short"],
            score=0.2,
            novelty_score=0.1,
            quality_score=0.3,
        )
        assert abs(result.novelty_score - 0.1) < 1e-9
        assert abs(result.quality_score - 0.3) < 1e-9


class TestContextAwareWriteGate:
    """Tests for ContextAwareWriteGate."""

    @pytest.mark.asyncio
    async def test_high_importance_override(self):
        """High importance memories are always accepted."""
        gate = ContextAwareWriteGate()
        memory = _make_memory(importance=0.9)  # Above default 0.8 threshold

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.ACCEPT
        assert "High importance override" in result.reasons

    @pytest.mark.asyncio
    async def test_semantic_auto_approval(self):
        """Semantic memories are auto-approved by default."""
        gate = ContextAwareWriteGate()
        memory = _make_memory(memory_type=MemoryType.SEMANTIC, importance=0.5)

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.ACCEPT
        assert "Semantic memory auto-approval" in result.reasons

    @pytest.mark.asyncio
    async def test_too_short_content_rejected(self):
        """Content below min length is rejected."""
        policy = WritePolicy(min_content_length=50)
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(content="Short")

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.REJECT
        assert RejectionReason.TOO_SHORT.value in result.reasons

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self):
        """Low confidence memories are rejected."""
        policy = WritePolicy(min_confidence=0.9, auto_approve_semantic=False)
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(confidence=0.5, memory_type=MemoryType.EPISODIC)

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.REJECT
        assert RejectionReason.LOW_CONFIDENCE.value in result.reasons

    @pytest.mark.asyncio
    async def test_memory_type_not_allowed(self):
        """Disallowed memory types are rejected."""
        policy = WritePolicy(memory_types_allowed=[MemoryType.SEMANTIC])
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(memory_type=MemoryType.EPISODIC)

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.REJECT
        assert "not allowed by policy" in result.reasons[0]

    @pytest.mark.asyncio
    async def test_novelty_assessment_with_no_existing(self):
        """Without existing memories, content is considered novel."""
        policy = WritePolicy(auto_approve_semantic=False)
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(memory_type=MemoryType.EPISODIC)

        result = await gate.evaluate(memory, existing_memories=None)

        # Should be accepted due to novelty
        assert result.decision in [WriteDecision.ACCEPT, WriteDecision.REQUEST_APPROVAL]

    @pytest.mark.asyncio
    async def test_pii_detection_email(self):
        """PII detection blocks emails when enabled."""
        policy = WritePolicy(
            block_pii=True,
            auto_approve_semantic=False,
            require_meaningful_content=False,  # Skip meaningfulness check
        )
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(
            content="Please contact me at user@example.com for more details about this project",
            memory_type=MemoryType.EPISODIC,
        )

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.REJECT
        assert any("email" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_pii_detection_phone(self):
        """PII detection blocks phone numbers when enabled."""
        policy = WritePolicy(
            block_pii=True,
            auto_approve_semantic=False,
            require_meaningful_content=False,  # Skip meaningfulness check
        )
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(
            content="You can call me at 555-123-4567 tomorrow for the meeting",
            memory_type=MemoryType.EPISODIC,
        )

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.REJECT
        assert any("phone" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_noise_detection_too_few_unique_words(self):
        """Content with too few unique words is rejected as noise."""
        policy = WritePolicy(require_meaningful_content=True, auto_approve_semantic=False)
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(
            content="test test test test test test",
            memory_type=MemoryType.EPISODIC,
        )

        result = await gate.evaluate(memory)

        assert result.decision == WriteDecision.REJECT
        assert RejectionReason.NOISE.value in result.reasons

    @pytest.mark.asyncio
    async def test_context_relevance_scoring(self):
        """Context hints improve relevance scoring."""
        policy = WritePolicy(auto_approve_semantic=False)
        gate = ContextAwareWriteGate(policy=policy)
        memory = _make_memory(
            content="The project meeting was very productive and insightful",
            memory_type=MemoryType.EPISODIC,
        )
        context = {
            "user_query": "Tell me about the project",
            "conversation_id": "conv-123",
        }

        result = await gate.evaluate(memory, context=context)

        # Relevance should be boosted
        assert result.relevance_score > 0.5

    @pytest.mark.asyncio
    async def test_custom_novelty_assessor(self):
        """Custom novelty assessor is used when provided."""
        custom_assessor = AsyncMock(return_value=0.9)
        policy = WritePolicy(auto_approve_semantic=False)
        gate = ContextAwareWriteGate(policy=policy, novelty_assessor=custom_assessor)
        memory = _make_memory(memory_type=MemoryType.EPISODIC)
        existing = [_make_memory()]

        result = await gate.evaluate(memory, existing_memories=existing)

        custom_assessor.assert_called_once()
        assert result.novelty_score is not None


class TestWriteGateCosimeSimilarity:
    """Tests for cosine similarity helper."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 0.0, 0.0]
        similarity = ContextAwareWriteGate._cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.01

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = ContextAwareWriteGate._cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.01

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        similarity = ContextAwareWriteGate._cosine_similarity(vec1, vec2)
        assert abs(similarity + 1.0) < 0.01
