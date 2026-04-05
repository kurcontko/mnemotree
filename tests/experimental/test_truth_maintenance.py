"""Tests for the truth maintenance / claims registry system."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.experimental.truth_maintenance import (
    Claim,
    ClaimsRegistry,
    ClaimStatus,
    ConflictSeverity,
    ResolutionStrategy,
)


def _memory(content: str, *, memory_id: str = "m1") -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
    )


class TestClaimsRegistry:
    @pytest.mark.asyncio
    async def test_extract_claims_no_llm(self):
        """Without LLM, should create a single fallback claim."""
        registry = ClaimsRegistry(llm=None)
        memory = _memory("Python is a programming language")
        claims = await registry.extract_claims(memory)
        assert len(claims) == 1
        assert claims[0].subject == "memory"
        assert claims[0].predicate == "contains"
        assert "Python" in claims[0].statement

    @pytest.mark.asyncio
    async def test_register_claims(self):
        registry = ClaimsRegistry(llm=None)
        claim = Claim(
            subject="python",
            predicate="version",
            object="3.12",
            statement="Python version is 3.12",
        )
        ids = await registry.register_claims([claim])
        assert len(ids) == 1
        assert ids[0] == claim.claim_id
        assert claim.claim_id in registry.claims

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        """Two claims with same subject+predicate but different objects should conflict."""
        registry = ClaimsRegistry(llm=None)
        claim1 = Claim(
            subject="python",
            predicate="version",
            object="3.11",
            statement="Python version is 3.11",
        )
        claim2 = Claim(
            subject="python",
            predicate="version",
            object="3.12",
            statement="Python version is 3.12",
        )
        await registry.register_claims([claim1])
        await registry.register_claims([claim2])
        assert len(registry.conflicts) == 1
        conflict = list(registry.conflicts.values())[0]
        assert set(conflict.claim_ids) == {claim1.claim_id, claim2.claim_id}
        assert conflict.severity == ConflictSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_no_conflict_same_object(self):
        """Claims with same subject+predicate+object should not conflict."""
        registry = ClaimsRegistry(llm=None)
        claim1 = Claim(subject="python", predicate="version", object="3.12", statement="a")
        claim2 = Claim(subject="python", predicate="version", object="3.12", statement="b")
        await registry.register_claims([claim1])
        await registry.register_claims([claim2])
        assert len(registry.conflicts) == 0

    @pytest.mark.asyncio
    async def test_no_conflict_different_subject(self):
        """Claims about different subjects should not conflict."""
        registry = ClaimsRegistry(llm=None)
        claim1 = Claim(subject="python", predicate="version", object="3.12", statement="a")
        claim2 = Claim(subject="rust", predicate="version", object="1.80", statement="b")
        await registry.register_claims([claim1])
        await registry.register_claims([claim2])
        assert len(registry.conflicts) == 0

    @pytest.mark.asyncio
    async def test_resolve_most_recent(self):
        registry = ClaimsRegistry(llm=None)
        old = Claim(
            subject="x",
            predicate="is",
            object="1",
            statement="x is 1",
            updated_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        new = Claim(
            subject="x",
            predicate="is",
            object="2",
            statement="x is 2",
            updated_at=datetime.now(timezone.utc),
        )
        await registry.register_claims([old])
        await registry.register_claims([new])

        conflict_id = list(registry.conflicts.keys())[0]
        winner_id = await registry.resolve_conflict(conflict_id, ResolutionStrategy.MOST_RECENT)
        assert winner_id == new.claim_id
        assert old.status == ClaimStatus.SUPERSEDED

    @pytest.mark.asyncio
    async def test_resolve_highest_confidence(self):
        registry = ClaimsRegistry(llm=None)
        low = Claim(subject="x", predicate="is", object="1", statement="a", confidence=0.3)
        high = Claim(subject="x", predicate="is", object="2", statement="b", confidence=0.9)
        await registry.register_claims([low])
        await registry.register_claims([high])

        conflict_id = list(registry.conflicts.keys())[0]
        winner_id = await registry.resolve_conflict(
            conflict_id, ResolutionStrategy.HIGHEST_CONFIDENCE
        )
        assert winner_id == high.claim_id

    @pytest.mark.asyncio
    async def test_resolve_ensemble(self):
        registry = ClaimsRegistry(llm=None)
        c1 = Claim(
            subject="x",
            predicate="is",
            object="1",
            statement="a",
            confidence=0.9,
            access_count=10,
        )
        c2 = Claim(subject="x", predicate="is", object="2", statement="b", confidence=0.1)
        await registry.register_claims([c1])
        await registry.register_claims([c2])

        conflict_id = list(registry.conflicts.keys())[0]
        winner_id = await registry.resolve_conflict(conflict_id, ResolutionStrategy.ENSEMBLE)
        assert winner_id == c1.claim_id

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_conflict(self):
        registry = ClaimsRegistry(llm=None)
        result = await registry.resolve_conflict("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_already_resolved(self):
        registry = ClaimsRegistry(llm=None)
        c1 = Claim(subject="x", predicate="is", object="1", statement="a")
        c2 = Claim(subject="x", predicate="is", object="2", statement="b")
        await registry.register_claims([c1])
        await registry.register_claims([c2])

        conflict_id = list(registry.conflicts.keys())[0]
        await registry.resolve_conflict(conflict_id)
        # Resolving again should return None
        result = await registry.resolve_conflict(conflict_id)
        assert result is None

    def test_check_staleness(self):
        registry = ClaimsRegistry(llm=None, staleness_threshold_days=30)
        old_claim = Claim(
            subject="x",
            predicate="is",
            object="1",
            statement="old",
            updated_at=datetime.now(timezone.utc) - timedelta(days=60),
            access_count=2,
        )
        recent_claim = Claim(
            subject="y",
            predicate="is",
            object="2",
            statement="recent",
            updated_at=datetime.now(timezone.utc),
        )
        registry.claims[old_claim.claim_id] = old_claim
        registry.claims[recent_claim.claim_id] = recent_claim
        stale = registry.check_staleness()
        assert old_claim.claim_id in stale
        assert recent_claim.claim_id not in stale

    def test_check_staleness_high_access_not_stale(self):
        """Claims with high access count should not be considered stale."""
        registry = ClaimsRegistry(llm=None, staleness_threshold_days=30)
        old_but_used = Claim(
            subject="x",
            predicate="is",
            object="1",
            statement="old but used",
            updated_at=datetime.now(timezone.utc) - timedelta(days=60),
            access_count=10,
        )
        registry.claims[old_but_used.claim_id] = old_but_used
        assert old_but_used.claim_id not in registry.check_staleness()

    @pytest.mark.asyncio
    async def test_register_empty_claims_list(self):
        registry = ClaimsRegistry(llm=None)
        ids = await registry.register_claims([])
        assert ids == []
