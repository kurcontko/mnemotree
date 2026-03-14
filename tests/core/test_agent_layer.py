"""Tests for agent layer phases 2, 5, 6, 7, 8."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core.memory import (
    AgentLayerConfig,
    MemoryCore,
    RecallFilters,
    RememberOptions,
)
from mnemotree.core.models import (
    MemoryItem,
    MemoryType,
    ObservationKind,
    ObservationStatus,
)


def _make_memory(
    memory_id: str = "mem-1",
    content: str = "test content",
    importance: float = 0.5,
    observation_status: ObservationStatus | None = None,
    observation_kind: ObservationKind | None = None,
    evidence_refs: list[str] | None = None,
    is_hot: bool = False,
    timestamp: datetime | None = None,
    repo_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        observation_status=observation_status,
        observation_kind=observation_kind,
        evidence_refs=evidence_refs or [],
        is_hot=is_hot,
        timestamp=timestamp or datetime.now(timezone.utc),
        repo_id=repo_id,
        task_id=task_id,
        agent_id=agent_id,
    )


# ============================================================
# Phase 2: Observation Semantics
# ============================================================


class TestObservationSemantics:
    def test_observation_status_enum_values(self):
        assert ObservationStatus.HYPOTHESIS.value == "hypothesis"
        assert ObservationStatus.TENTATIVE.value == "tentative"
        assert ObservationStatus.CONFIRMED.value == "confirmed"
        assert ObservationStatus.REFUTED.value == "refuted"

    def test_observation_kind_enum_values(self):
        assert ObservationKind.ATTEMPT.value == "attempt"
        assert ObservationKind.RESULT.value == "result"
        assert ObservationKind.DECISION.value == "decision"
        assert ObservationKind.HANDOFF.value == "handoff"
        assert ObservationKind.WARNING.value == "warning"
        assert ObservationKind.OBSERVATION.value == "observation"

    def test_memory_item_observation_fields(self):
        mem = _make_memory(
            observation_status=ObservationStatus.TENTATIVE,
            observation_kind=ObservationKind.ATTEMPT,
            evidence_refs=["abc123", "tests/test_foo.py"],
        )
        assert mem.observation_status == ObservationStatus.TENTATIVE
        assert mem.observation_kind == ObservationKind.ATTEMPT
        assert mem.evidence_refs == ["abc123", "tests/test_foo.py"]

    def test_memory_item_observation_defaults(self):
        mem = _make_memory()
        assert mem.observation_status is None
        assert mem.observation_kind is None
        assert mem.evidence_refs == []

    def test_observation_status_serialization(self):
        mem = _make_memory(observation_status=ObservationStatus.CONFIRMED)
        data = mem.model_dump(mode="json")
        assert data["observation_status"] == "confirmed"

    def test_observation_kind_serialization(self):
        mem = _make_memory(observation_kind=ObservationKind.DECISION)
        data = mem.model_dump(mode="json")
        assert data["observation_kind"] == "decision"

    def test_evidence_refs_serialization(self):
        refs = ["commit:abc123", "/path/to/file.py", "https://github.com/issue/1"]
        mem = _make_memory(evidence_refs=refs)
        data = mem.model_dump(mode="json")
        assert data["evidence_refs"] == refs


class TestRecallFiltersObservation:
    def test_recall_filters_observation_status(self):
        f = RecallFilters(
            observation_status=[ObservationStatus.CONFIRMED, ObservationStatus.TENTATIVE]
        )
        assert len(f.observation_status) == 2

    def test_recall_filters_exclude_refuted(self):
        f = RecallFilters(exclude_refuted=True)
        assert f.exclude_refuted is True

    def test_recall_filters_observation_kind(self):
        f = RecallFilters(observation_kind=[ObservationKind.DECISION])
        assert f.observation_kind == [ObservationKind.DECISION]


class TestRememberOptionsObservation:
    def test_remember_options_observation_fields(self):
        opts = RememberOptions(
            observation_status=ObservationStatus.HYPOTHESIS,
            observation_kind=ObservationKind.ATTEMPT,
            evidence_refs=["test:unit_test_1"],
        )
        assert opts.observation_status == ObservationStatus.HYPOTHESIS
        assert opts.observation_kind == ObservationKind.ATTEMPT
        assert opts.evidence_refs == ["test:unit_test_1"]


# ============================================================
# Phase 5: Summary-First Retrieval (tested via MCP in test_server)
# ============================================================


# ============================================================
# Phase 6: Validation and Freshness
# ============================================================


class TestFreshnessScoring:
    def _make_core(self, **agent_cfg_kwargs) -> MemoryCore:
        store = MagicMock()
        store.close = AsyncMock()
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 384)
        agent_config = AgentLayerConfig(**agent_cfg_kwargs)
        return MemoryCore(
            store=store,
            embeddings=embeddings,
            agent_layer_config=agent_config,
        )

    def test_fresh_memories_not_penalized(self):
        core = self._make_core(freshness_staleness_hours=168.0, freshness_penalty_factor=0.3)
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory("m1", importance=0.8, timestamp=now - timedelta(hours=1)),
            _make_memory("m2", importance=0.7, timestamp=now - timedelta(hours=2)),
        ]
        result = core.apply_freshness_scoring(memories)
        assert result[0].memory_id == "m1"  # Higher importance first
        assert result[1].memory_id == "m2"

    def test_stale_memories_downranked(self):
        core = self._make_core(freshness_staleness_hours=24.0, freshness_penalty_factor=0.5)
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory("old", importance=0.9, timestamp=now - timedelta(hours=72)),
            _make_memory("new", importance=0.6, timestamp=now - timedelta(hours=1)),
        ]
        result = core.apply_freshness_scoring(memories)
        # New memory should rank higher despite lower raw importance
        assert result[0].memory_id == "new"

    def test_refuted_memories_heavily_penalized(self):
        core = self._make_core()
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory("refuted", importance=0.9, observation_status=ObservationStatus.REFUTED, timestamp=now),
            _make_memory("normal", importance=0.5, timestamp=now),
        ]
        result = core.apply_freshness_scoring(memories)
        assert result[0].memory_id == "normal"

    def test_confirmed_memories_boosted(self):
        core = self._make_core()
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory("confirmed", importance=0.5, observation_status=ObservationStatus.CONFIRMED, timestamp=now),
            _make_memory("tentative", importance=0.5, observation_status=ObservationStatus.TENTATIVE, timestamp=now),
        ]
        result = core.apply_freshness_scoring(memories)
        assert result[0].memory_id == "confirmed"


class TestEvidenceValidation:
    def _make_core(self) -> MemoryCore:
        store = MagicMock()
        store.close = AsyncMock()
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 384)
        return MemoryCore(store=store, embeddings=embeddings)

    def test_file_path_validation(self, tmp_path):
        core = self._make_core()
        existing_file = tmp_path / "test.py"
        existing_file.write_text("pass")

        mem = _make_memory(evidence_refs=[str(existing_file), "/nonexistent/path"])
        results = core.validate_evidence_refs(mem)
        assert results[str(existing_file)] == "valid"
        assert results["/nonexistent/path"] == "invalid"

    def test_commit_sha_unverified(self):
        core = self._make_core()
        mem = _make_memory(evidence_refs=["abc1234567890"])
        results = core.validate_evidence_refs(mem)
        assert results["abc1234567890"] == "unverified"

    def test_url_unverified(self):
        core = self._make_core()
        mem = _make_memory(evidence_refs=["https://github.com/issue/123"])
        results = core.validate_evidence_refs(mem)
        assert results["https://github.com/issue/123"] == "unverified"

    def test_empty_evidence_refs(self):
        core = self._make_core()
        mem = _make_memory(evidence_refs=[])
        results = core.validate_evidence_refs(mem)
        assert results == {}


# ============================================================
# Phase 7: Agent Layer Config
# ============================================================


class TestAgentLayerConfig:
    def test_default_config(self):
        config = AgentLayerConfig()
        assert config.enable_agent_scoping is True
        assert config.enable_leases is True
        assert config.enable_summaries is True
        assert config.enable_observation_semantics is True
        assert config.enable_validation is False
        assert config.enable_freshness_scoring is False
        assert config.agent_recall_limit == 20
        assert config.summary_compaction_interval == 3600
        assert config.freshness_staleness_hours == 168.0
        assert config.freshness_penalty_factor == 0.3
        assert config.exclude_refuted_by_default is True

    def test_custom_config(self):
        config = AgentLayerConfig(
            enable_leases=False,
            enable_validation=True,
            agent_recall_limit=50,
            freshness_staleness_hours=48.0,
        )
        assert config.enable_leases is False
        assert config.enable_validation is True
        assert config.agent_recall_limit == 50
        assert config.freshness_staleness_hours == 48.0

    def test_memory_core_accepts_agent_config(self):
        store = MagicMock()
        store.close = AsyncMock()
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 384)
        config = AgentLayerConfig(enable_leases=False)
        core = MemoryCore(store=store, embeddings=embeddings, agent_layer_config=config)
        assert core._agent_layer_config.enable_leases is False

    def test_memory_core_default_agent_config(self):
        store = MagicMock()
        store.close = AsyncMock()
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 384)
        core = MemoryCore(store=store, embeddings=embeddings)
        assert core._agent_layer_config.enable_agent_scoping is True


# ============================================================
# Phase 2 + Store: SQLite record round-trip
# ============================================================


class TestSQLiteObservationRoundTrip:
    def test_sqlite_record_includes_observation_fields(self):
        from mnemotree.store._records import sqlite_record_from_memory

        mem = _make_memory(
            observation_status=ObservationStatus.CONFIRMED,
            observation_kind=ObservationKind.DECISION,
            evidence_refs=["abc123", "tests/foo.py"],
        )
        record = sqlite_record_from_memory(mem)
        assert record["observation_status"] == "confirmed"
        assert record["observation_kind"] == "decision"
        assert json.loads(record["evidence_refs"]) == ["abc123", "tests/foo.py"]

    def test_sqlite_record_none_observation_fields(self):
        from mnemotree.store._records import sqlite_record_from_memory

        mem = _make_memory()
        record = sqlite_record_from_memory(mem)
        assert record["observation_status"] is None
        assert record["observation_kind"] is None
        assert record["evidence_refs"] is None


# ============================================================
# Integration: Config enforcement in recall path
# ============================================================


class TestConfigEnforcementInRecall:
    """Test that AgentLayerConfig knobs are actually enforced during recall."""

    def _make_core(self, repo_id: str = "test-repo", **cfg_kwargs) -> MemoryCore:
        store = MagicMock()
        store.close = AsyncMock()
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 384)
        agent_config = AgentLayerConfig(**cfg_kwargs)
        return MemoryCore(
            store=store,
            embeddings=embeddings,
            agent_layer_config=agent_config,
            default_repo_id=repo_id,
        )

    @pytest.mark.asyncio
    async def test_exclude_refuted_by_default_applies_to_agent_scoped_recall(self):
        """When repo_id is set and exclude_refuted_by_default=True, refuted memories are excluded."""
        core = self._make_core(exclude_refuted_by_default=True, enable_observation_semantics=True)

        # Create memories with different statuses — must match the repo_id scope
        now = datetime.now(timezone.utc)
        confirmed = _make_memory("m1", observation_status=ObservationStatus.CONFIRMED, timestamp=now, repo_id="test-repo")
        refuted = _make_memory("m2", observation_status=ObservationStatus.REFUTED, timestamp=now, repo_id="test-repo")
        tentative = _make_memory("m3", observation_status=ObservationStatus.TENTATIVE, timestamp=now, repo_id="test-repo")

        # Mock the retrieval to return all three
        core.retrieval = MagicMock()
        core.retrieval.recall = AsyncMock(return_value=[confirmed, refuted, tentative])

        result = await core.recall("test query", limit=10)

        # Refuted should be filtered out
        result_ids = {m.memory_id for m in result}
        assert "m1" in result_ids
        assert "m2" not in result_ids  # refuted excluded
        assert "m3" in result_ids

    @pytest.mark.asyncio
    async def test_exclude_refuted_not_applied_without_agent_scope(self):
        """Without repo_id, exclude_refuted_by_default is NOT auto-applied."""
        store = MagicMock()
        store.close = AsyncMock()
        embeddings = MagicMock()
        embeddings.embed_query = MagicMock(return_value=[0.0] * 384)
        core = MemoryCore(
            store=store,
            embeddings=embeddings,
            agent_layer_config=AgentLayerConfig(exclude_refuted_by_default=True),
            # No default_repo_id!
        )

        now = datetime.now(timezone.utc)
        refuted = _make_memory("m1", observation_status=ObservationStatus.REFUTED, timestamp=now)

        core.retrieval = MagicMock()
        core.retrieval.recall = AsyncMock(return_value=[refuted])

        result = await core.recall("test query", limit=10)

        # Without agent scope, refuted should NOT be auto-excluded
        assert len(result) == 1
        assert result[0].memory_id == "m1"


# ============================================================
# MCP: Tool registration count
# ============================================================


def test_all_agent_tools_exist_as_module_functions():
    """Verify all expected agent MCP tool functions exist in the server module."""
    from mnemotree.mcp import server

    expected_tools = [
        "agent_remember_observation",
        "agent_recall_context",
        "agent_upsert_summary",
        "agent_get_summary",
        "agent_claim_resource",
        "agent_renew_claim",
        "agent_release_claim",
        "agent_list_claims",
        "agent_update_observation_status",
        "agent_recall_with_summary",
        "agent_compact_summary",
        "agent_inspect_memories",
        "agent_inspect_summaries",
        "agent_delete_memory",
        "agent_correct_memory",
    ]
    for tool_name in expected_tools:
        assert hasattr(server, tool_name), f"Missing MCP tool function: {tool_name}"
        assert callable(getattr(server, tool_name)), f"{tool_name} is not callable"
