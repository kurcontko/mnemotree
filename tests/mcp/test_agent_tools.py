"""Tests for agent layer MCP tools (phases 2, 5, 7, 8)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemotree.core.models import MemoryItem, MemoryType, ObservationKind, ObservationStatus
from mnemotree.mcp import server


def _make_memory(
    memory_id: str,
    ts: datetime | None = None,
    content: str = "test",
    observation_status: ObservationStatus | None = None,
    observation_kind: ObservationKind | None = None,
    evidence_refs: list[str] | None = None,
    is_hot: bool = False,
    repo_id: str | None = "repo-1",
    task_id: str | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
        observation_status=observation_status,
        observation_kind=observation_kind,
        evidence_refs=evidence_refs or [],
        is_hot=is_hot,
        repo_id=repo_id,
        task_id=task_id,
    )


# ============================================================
# Phase 2: agent_remember_observation with evidence_refs
# ============================================================


@pytest.mark.asyncio
async def test_agent_remember_observation_with_evidence(monkeypatch):
    memory = _make_memory("mem-1")
    memory_core = MagicMock()
    memory_core.remember = AsyncMock(return_value=memory)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    await server.agent_remember_observation(
        repo_id="repo-1",
        content="Tests pass after fix",
        kind="result",
        observation_status="confirmed",
        evidence_refs=["abc123", "tests/test_fix.py"],
    )

    kwargs = memory_core.remember.await_args.kwargs
    assert kwargs["metadata"]["observation_status"] == "confirmed"
    assert kwargs["metadata"]["observation_kind"] == "result"
    assert kwargs["metadata"]["evidence_refs"] == ["abc123", "tests/test_fix.py"]


@pytest.mark.asyncio
async def test_agent_remember_observation_default_kind(monkeypatch):
    memory = _make_memory("mem-1")
    memory_core = MagicMock()
    memory_core.remember = AsyncMock(return_value=memory)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    await server.agent_remember_observation(
        repo_id="repo-1",
        content="Something happened",
    )

    kwargs = memory_core.remember.await_args.kwargs
    assert kwargs["metadata"]["observation_kind"] == "observation"
    assert kwargs["metadata"]["observation_status"] == "tentative"


# ============================================================
# Phase 2: agent_update_observation_status
# ============================================================


@pytest.mark.asyncio
async def test_agent_update_observation_status(monkeypatch):
    memory = _make_memory(
        "mem-1",
        observation_status=ObservationStatus.TENTATIVE,
        evidence_refs=["old-ref"],
    )
    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.update_memory_metadata = AsyncMock(return_value=True)

    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    result = await server.agent_update_observation_status(
        repo_id="repo-1",
        memory_id="mem-1",
        observation_status="confirmed",
        evidence_refs=["new-ref"],
    )

    assert result["updated"] is True
    assert result["observation_status"] == "confirmed"
    call_kwargs = store.update_memory_metadata.await_args
    updates = call_kwargs[0][1]
    assert updates["observation_status"] == "confirmed"
    assert "old-ref" in updates["evidence_refs"]
    assert "new-ref" in updates["evidence_refs"]


@pytest.mark.asyncio
async def test_agent_update_observation_status_invalid(monkeypatch):
    memory = _make_memory("mem-1")
    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="Invalid observation_status"):
        await server.agent_update_observation_status(
            repo_id="repo-1",
            memory_id="mem-1",
            observation_status="bogus",
        )


@pytest.mark.asyncio
async def test_agent_update_observation_status_not_found(monkeypatch):
    store = MagicMock()
    store.get_memory = AsyncMock(return_value=None)
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="not found"):
        await server.agent_update_observation_status(
            repo_id="repo-1",
            memory_id="missing",
            observation_status="confirmed",
        )


# ============================================================
# Phase 5: agent_recall_with_summary
# ============================================================


@pytest.mark.asyncio
async def test_agent_recall_with_summary(monkeypatch):
    memory_core = MagicMock()
    store = MagicMock()
    store.get_summary = AsyncMock(return_value={
        "content": "Task progress: 50% done",
        "scope_kind": "task",
    })
    store.upsert_summary = AsyncMock()
    memory_core.store = store

    # Mock recall to return serialized dicts
    memories = [
        {
            "memory_id": "m1",
            "content": "Found the bug",
            "summary": "Found the bug",
            "is_hot": True,
            "observation_status": "confirmed",
            "metadata": {},
        },
        {
            "memory_id": "m2",
            "content": "Tried approach A",
            "summary": "Tried approach A",
            "is_hot": False,
            "observation_status": "tentative",
            "metadata": {},
        },
        {
            "memory_id": "m3",
            "content": "This was wrong",
            "summary": "This was wrong",
            "is_hot": False,
            "observation_status": "refuted",
            "metadata": {},
        },
    ]

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))
    monkeypatch.setattr(server, "recall", AsyncMock(return_value=memories))

    result = await server.agent_recall_with_summary(
        repo_id="repo-1",
        query="what happened?",
        task_id="task-1",
        exclude_refuted=True,
    )

    assert result["summary"]["content"] == "Task progress: 50% done"
    assert len(result["hot_observations"]) == 1
    assert result["hot_observations"][0]["memory_id"] == "m1"
    # m3 excluded because refuted
    assert all(m["memory_id"] != "m3" for m in result["observations"])


@pytest.mark.asyncio
async def test_agent_recall_with_summary_no_summary(monkeypatch):
    memory_core = MagicMock()
    store = MagicMock()
    store.get_summary = AsyncMock(return_value=None)
    memory_core.store = store

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))
    monkeypatch.setattr(server, "recall", AsyncMock(return_value=[]))

    result = await server.agent_recall_with_summary(
        repo_id="repo-1",
        query="anything?",
    )
    assert result["summary"] is None
    assert result["total_results"] == 0


# ============================================================
# Phase 5: agent_compact_summary
# ============================================================


@pytest.mark.asyncio
async def test_agent_compact_summary(monkeypatch):
    store = MagicMock()
    store.get_summary = AsyncMock(return_value=None)
    store.upsert_summary = AsyncMock(return_value={
        "summary_id": "s1",
        "content": "compacted",
    })
    memory_core = MagicMock()
    memory_core.store = store

    memories = [
        {"memory_id": "m1", "content": "Did X", "summary": "Did X", "metadata": {"kind": "attempt"}},
        {"memory_id": "m2", "content": "Result Y", "summary": "Result Y", "metadata": {"observation_status": "confirmed"}},
    ]

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))
    monkeypatch.setattr(server, "recall", AsyncMock(return_value=memories))

    result = await server.agent_compact_summary(
        repo_id="repo-1",
        scope_kind="task",
        task_id="task-1",
    )

    assert result["compacted"] is True
    assert result["observation_count"] == 2
    store.upsert_summary.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_compact_summary_empty(monkeypatch):
    store = MagicMock()
    store.get_summary = AsyncMock(return_value=None)
    store.upsert_summary = AsyncMock()
    memory_core = MagicMock()
    memory_core.store = store

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))
    monkeypatch.setattr(server, "recall", AsyncMock(return_value=[]))

    result = await server.agent_compact_summary(
        repo_id="repo-1",
        scope_kind="task",
    )

    assert result["compacted"] is False


# ============================================================
# Phase 8: Human Surface
# ============================================================


@pytest.mark.asyncio
async def test_agent_inspect_memories(monkeypatch):
    memories = [
        _make_memory("m1", repo_id="repo-1", task_id="t1",
                      observation_status=ObservationStatus.CONFIRMED,
                      observation_kind=ObservationKind.DECISION),
        _make_memory("m2", repo_id="repo-1", task_id="t1",
                      observation_status=ObservationStatus.REFUTED),
        _make_memory("m3", repo_id="repo-2"),  # different repo
    ]

    store = MagicMock()
    store.list_memories = AsyncMock(return_value=memories)
    memory_core = MagicMock()
    memory_core.store = store

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    result = await server.agent_inspect_memories(
        repo_id="repo-1",
        task_id="t1",
    )

    assert result["count"] == 2
    assert result["memories"][0]["memory_id"] == "m1"
    assert result["memories"][0]["observation_status"] == "confirmed"
    assert result["memories"][0]["observation_kind"] == "decision"


@pytest.mark.asyncio
async def test_agent_inspect_memories_filter_by_status(monkeypatch):
    memories = [
        _make_memory("m1", repo_id="repo-1",
                      observation_status=ObservationStatus.CONFIRMED),
        _make_memory("m2", repo_id="repo-1",
                      observation_status=ObservationStatus.REFUTED),
    ]

    store = MagicMock()
    store.list_memories = AsyncMock(return_value=memories)
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    result = await server.agent_inspect_memories(
        repo_id="repo-1",
        observation_status="confirmed",
    )

    assert result["count"] == 1
    assert result["memories"][0]["memory_id"] == "m1"


@pytest.mark.asyncio
async def test_agent_inspect_summaries(monkeypatch):
    store = MagicMock()
    store.upsert_summary = AsyncMock()
    store.get_summary = AsyncMock(side_effect=[
        {"scope_kind": "repo", "content": "Repo summary"},
        None,  # worktree
        {"scope_kind": "task", "content": "Task summary"},
        None,  # sprint
        None,  # feature
    ])
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    result = await server.agent_inspect_summaries(repo_id="repo-1")

    assert result["count"] == 2


@pytest.mark.asyncio
async def test_agent_delete_memory(monkeypatch):
    store = MagicMock()
    store.delete_memory = AsyncMock(return_value=True)
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    result = await server.agent_delete_memory(repo_id="repo-1", memory_id="mem-1")

    assert result["deleted"] is True
    store.delete_memory.assert_awaited_once_with("mem-1")


@pytest.mark.asyncio
async def test_agent_correct_memory(monkeypatch):
    memory = _make_memory("mem-1", observation_status=ObservationStatus.TENTATIVE)
    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.update_memory_metadata = AsyncMock(return_value=True)
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    result = await server.agent_correct_memory(
        repo_id="repo-1",
        memory_id="mem-1",
        content="corrected content",
        observation_status="confirmed",
        importance=0.9,
    )

    assert "content" in result["updated_fields"]
    assert "observation_status" in result["updated_fields"]
    assert "importance" in result["updated_fields"]


@pytest.mark.asyncio
async def test_agent_correct_memory_not_found(monkeypatch):
    store = MagicMock()
    store.get_memory = AsyncMock(return_value=None)
    memory_core = MagicMock()
    memory_core.store = store
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="not found"):
        await server.agent_correct_memory(
            repo_id="repo-1",
            memory_id="missing",
            content="nope",
        )


# ============================================================
# Phase 8: Compact index includes observation fields
# ============================================================


def test_serialize_memory_index_includes_observation_fields():
    mem = _make_memory(
        "m1",
        observation_status=ObservationStatus.CONFIRMED,
        observation_kind=ObservationKind.DECISION,
        is_hot=True,
    )
    result = server._serialize_memory_index(mem, 1)
    assert result["observation_status"] == "confirmed"
    assert result["observation_kind"] == "decision"
    assert result["is_hot"] is True


def test_serialize_memory_index_none_observation_fields():
    mem = _make_memory("m1")
    result = server._serialize_memory_index(mem, 1)
    assert result["observation_status"] is None
    assert result["observation_kind"] is None
    assert result["is_hot"] is False
