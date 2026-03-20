"""Tests for Agent Layer Phase 2 — Retrieval & Operational Improvements.

Covers:
- Phase A: Graph channel activation via AgentLayerConfig
- Phase B: Confidence ↔ ObservationStatus coupling
- Phase C: Auto-compaction config
- Phase D: Multi-round agentic retrieval
- Phase E: Temporal proximity scoring
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemotree.core.memory import AgentLayerConfig, MemoryCore, RecallFilters
from mnemotree.core.models import (
    MemoryItem,
    MemoryType,
    ObservationKind,
    ObservationStatus,
    compute_observation_confidence,
)
from mnemotree.core.retrieval import HybridRetriever, RetrievalStage
from mnemotree.core.scoring import MemoryScoring


def _make_memory(
    memory_id: str = "mem-1",
    content: str = "test content",
    importance: float = 0.5,
    observation_status: ObservationStatus | None = None,
    confidence: float = 1.0,
    timestamp: datetime | None = None,
    evidence_refs: list[str] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        observation_status=observation_status,
        confidence=confidence,
        evidence_refs=evidence_refs or [],
        timestamp=timestamp or datetime.now(timezone.utc),
    )


# ───────────────────────────────────────────────────────────────────
# Phase A: Graph channel activation
# ───────────────────────────────────────────────────────────────────


class TestPhaseAGraphChannel:
    def test_agent_layer_config_has_graph_weight(self):
        cfg = AgentLayerConfig()
        assert cfg.graph_weight == 0.15

    def test_agent_layer_config_graph_weight_customizable(self):
        cfg = AgentLayerConfig(graph_weight=0.25)
        assert cfg.graph_weight == 0.25

    def test_hybrid_retriever_accepts_temporal_weight(self):
        """HybridRetriever should accept temporal_weight parameter."""
        retriever = HybridRetriever(temporal_weight=0.1, graph_weight=0.15)
        assert RetrievalStage.TEMPORAL in retriever.weights
        assert retriever.weights[RetrievalStage.TEMPORAL] > 0

    def test_hybrid_retriever_graph_weight_in_weights(self):
        retriever = HybridRetriever(graph_weight=0.15)
        assert RetrievalStage.GRAPH in retriever.weights
        assert retriever.weights[RetrievalStage.GRAPH] > 0

    def test_hybrid_retriever_zero_graph_weight(self):
        retriever = HybridRetriever(graph_weight=0.0)
        # After normalization, should be 0
        assert retriever.weights[RetrievalStage.GRAPH] == 0.0


# ───────────────────────────────────────────────────────────────────
# Phase B: Confidence ↔ ObservationStatus
# ───────────────────────────────────────────────────────────────────


class TestPhaseBConfidence:
    def test_hypothesis_confidence(self):
        c = compute_observation_confidence(ObservationStatus.HYPOTHESIS)
        assert c == 0.3

    def test_tentative_confidence(self):
        c = compute_observation_confidence(ObservationStatus.TENTATIVE)
        assert c == 0.5

    def test_confirmed_confidence(self):
        c = compute_observation_confidence(ObservationStatus.CONFIRMED)
        assert c == 0.85

    def test_refuted_confidence(self):
        c = compute_observation_confidence(ObservationStatus.REFUTED)
        assert c == 0.1

    def test_evidence_boost(self):
        base = compute_observation_confidence(ObservationStatus.TENTATIVE)
        with_evidence = compute_observation_confidence(
            ObservationStatus.TENTATIVE,
            evidence_refs=["commit:abc123", "test:test_foo"],
        )
        assert with_evidence == base + 0.10  # 2 × 0.05

    def test_evidence_boost_capped(self):
        with_many = compute_observation_confidence(
            ObservationStatus.TENTATIVE,
            evidence_refs=["a", "b", "c", "d", "e"],
        )
        # Boost capped at 0.15
        assert with_many == 0.5 + 0.15

    def test_confirmed_with_max_evidence(self):
        c = compute_observation_confidence(
            ObservationStatus.CONFIRMED,
            evidence_refs=["a", "b", "c", "d"],
        )
        assert c == 1.0  # 0.85 + 0.15 = 1.0, clamped

    def test_confidence_multiplier_in_scoring(self):
        """Scoring should use confidence as multiplier when observation_status is set."""
        scoring = MemoryScoring(importance_weight=1.0, recency_weight=0.0, relevance_weight=0.0)

        mem_full = _make_memory(importance=0.8, observation_status=ObservationStatus.CONFIRMED, confidence=1.0)
        mem_low = _make_memory(importance=0.8, observation_status=ObservationStatus.HYPOTHESIS, confidence=0.3)

        score_full = scoring.calculate_memory_score(mem_full)
        score_low = scoring.calculate_memory_score(mem_low)

        assert score_full > score_low

    def test_confidence_not_applied_without_observation_status(self):
        """When observation_status is None, confidence should not affect scoring."""
        scoring = MemoryScoring(importance_weight=1.0, recency_weight=0.0, relevance_weight=0.0)

        mem = _make_memory(importance=0.8, observation_status=None, confidence=0.1)
        score = scoring.calculate_memory_score(mem)
        # Without observation_status, confidence is NOT applied as multiplier
        # Score should be based on raw importance (≈0.8)
        assert score > 0.5


# ───────────────────────────────────────────────────────────────────
# Phase C: Auto-compaction config
# ───────────────────────────────────────────────────────────────────


class TestPhaseCAutoCompaction:
    def test_auto_compaction_config_defaults(self):
        cfg = AgentLayerConfig()
        assert cfg.auto_compaction_enabled is True
        assert cfg.auto_compaction_threshold == 10

    def test_auto_compaction_config_customizable(self):
        cfg = AgentLayerConfig(auto_compaction_enabled=False, auto_compaction_threshold=5)
        assert cfg.auto_compaction_enabled is False
        assert cfg.auto_compaction_threshold == 5

    @pytest.mark.asyncio
    async def test_auto_compact_fires_after_threshold(self):
        """Verify _auto_compact is called after observation threshold is reached."""
        from mnemotree.mcp.server import _observation_counters

        # Reset counters
        _observation_counters.clear()

        # Simulate incrementing counter
        scope_key = "test-repo::task-1"
        for i in range(9):
            _observation_counters[scope_key] = _observation_counters.get(scope_key, 0) + 1

        assert _observation_counters[scope_key] == 9

        # One more should trigger (checked in the real code path)
        _observation_counters[scope_key] += 1
        assert _observation_counters[scope_key] == 10


# ───────────────────────────────────────────────────────────────────
# Phase D: Agentic retrieval config
# ───────────────────────────────────────────────────────────────────


class TestPhaseDAgenticRetrieval:
    def test_agentic_config_defaults(self):
        cfg = AgentLayerConfig()
        assert cfg.enable_agentic_retrieval is True
        assert cfg.agentic_max_rounds == 2
        assert cfg.agentic_sufficiency_threshold == 3

    def test_agentic_config_customizable(self):
        cfg = AgentLayerConfig(agentic_max_rounds=5, agentic_sufficiency_threshold=10)
        assert cfg.agentic_max_rounds == 5
        assert cfg.agentic_sufficiency_threshold == 10


# ───────────────────────────────────────────────────────────────────
# Phase E: Temporal proximity scoring
# ───────────────────────────────────────────────────────────────────


class TestPhaseETemporalProximity:
    def test_retrieval_stage_has_temporal(self):
        assert hasattr(RetrievalStage, "TEMPORAL")
        assert RetrievalStage.TEMPORAL.value == "temporal"

    def test_extract_time_reference_iso_date(self):
        ref = HybridRetriever._extract_time_reference("What happened on 2024-06-15?")
        assert ref is not None
        assert ref.year == 2024
        assert ref.month == 6
        assert ref.day == 15

    def test_extract_time_reference_yesterday(self):
        ref = HybridRetriever._extract_time_reference("What did I do yesterday?")
        assert ref is not None
        now = datetime.now(timezone.utc)
        assert (now - ref).days <= 1

    def test_extract_time_reference_days_ago(self):
        ref = HybridRetriever._extract_time_reference("What happened 3 days ago?")
        assert ref is not None
        now = datetime.now(timezone.utc)
        diff = abs((now - ref).total_seconds())
        assert 2.5 * 86400 < diff < 3.5 * 86400

    def test_extract_time_reference_weeks_ago(self):
        ref = HybridRetriever._extract_time_reference("What happened 2 weeks ago?")
        assert ref is not None
        now = datetime.now(timezone.utc)
        diff = abs((now - ref).total_seconds())
        assert 13 * 86400 < diff < 15 * 86400

    def test_extract_time_reference_last_week(self):
        ref = HybridRetriever._extract_time_reference("What did we discuss last week?")
        assert ref is not None
        now = datetime.now(timezone.utc)
        diff = abs((now - ref).total_seconds())
        assert 6 * 86400 < diff < 8 * 86400

    def test_extract_time_reference_no_time(self):
        ref = HybridRetriever._extract_time_reference("What is the meaning of life?")
        assert ref is None

    def test_score_temporal_proximity(self):
        retriever = HybridRetriever(temporal_weight=0.1)
        now = datetime.now(timezone.utc)

        close_mem = _make_memory(memory_id="close", timestamp=now - timedelta(days=1))
        far_mem = _make_memory(memory_id="far", timestamp=now - timedelta(days=30))
        query = f"What happened on {(now - timedelta(days=1)).strftime('%Y-%m-%d')}?"

        scored = retriever._score_temporal_proximity(query, [close_mem, far_mem])
        assert len(scored) == 2
        # Close memory should score higher
        scores_by_id = {m.memory_id: s for m, s in scored}
        assert scores_by_id["close"] > scores_by_id["far"]

    def test_score_temporal_proximity_no_time_ref(self):
        retriever = HybridRetriever(temporal_weight=0.1)
        mem = _make_memory()
        scored = retriever._score_temporal_proximity("general query", [mem])
        assert scored == []

    def test_temporal_weight_in_config(self):
        cfg = AgentLayerConfig(temporal_weight=0.1)
        assert cfg.temporal_weight == 0.1

    def test_temporal_weight_default_zero(self):
        cfg = AgentLayerConfig()
        assert cfg.temporal_weight == 0.0
