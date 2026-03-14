"""Tests for agent scoping fields (Phase 1)."""

from __future__ import annotations

from mnemotree.core.memory import RecallFilters, RememberOptions
from mnemotree.core.models import MemoryItem, MemoryType


class TestMemoryItemScopeFields:
    """Test scope fields on MemoryItem."""

    def test_default_scope_fields_are_none(self):
        item = MemoryItem(content="test", memory_type=MemoryType.SEMANTIC, importance=0.5)
        assert item.repo_id is None
        assert item.worktree_id is None
        assert item.task_id is None
        assert item.agent_id is None
        assert item.run_id is None

    def test_scope_fields_set_on_creation(self):
        item = MemoryItem(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            repo_id="myrepo",
            worktree_id="wt-1",
            task_id="task-42",
            agent_id="agent-a",
            run_id="run-xyz",
        )
        assert item.repo_id == "myrepo"
        assert item.worktree_id == "wt-1"
        assert item.task_id == "task-42"
        assert item.agent_id == "agent-a"
        assert item.run_id == "run-xyz"


class TestRememberOptionsScopeFields:
    """Test scope fields on RememberOptions."""

    def test_default_scope_fields_are_none(self):
        opts = RememberOptions()
        assert opts.repo_id is None
        assert opts.worktree_id is None
        assert opts.task_id is None
        assert opts.agent_id is None
        assert opts.run_id is None

    def test_scope_fields_set(self):
        opts = RememberOptions(repo_id="r", worktree_id="w", task_id="t")
        assert opts.repo_id == "r"
        assert opts.worktree_id == "w"
        assert opts.task_id == "t"


class TestRecallFiltersScopeFields:
    """Test scope fields on RecallFilters."""

    def test_default_scope_fields_are_none(self):
        filters = RecallFilters()
        assert filters.repo_id is None
        assert filters.worktree_id is None
        assert filters.task_id is None
        assert filters.agent_id is None
        assert filters.run_id is None

    def test_scope_fields_filter(self):
        filters = RecallFilters(repo_id="myrepo", agent_id="agent-a")
        assert filters.repo_id == "myrepo"
        assert filters.agent_id == "agent-a"


class TestRecallFiltersApply:
    """Test that _apply_recall_filters respects scope fields."""

    def _make_memory(self, **kwargs) -> MemoryItem:
        defaults = {
            "content": "test",
            "memory_type": MemoryType.SEMANTIC,
            "importance": 0.5,
        }
        defaults.update(kwargs)
        return MemoryItem(**defaults)

    def _apply_filters(self, memories, filters):
        """Call _apply_recall_filters via a minimal mock MemoryCore."""
        from unittest.mock import MagicMock

        from mnemotree.core.memory import MemoryCore

        mock_core = MagicMock(spec=MemoryCore)
        mock_core._coerce_filter_timestamp = MagicMock(return_value=None)
        return MemoryCore._apply_recall_filters(mock_core, memories, filters)

    def test_filter_by_repo_id(self):
        m1 = self._make_memory(repo_id="repo-a")
        m2 = self._make_memory(repo_id="repo-b")
        m3 = self._make_memory(repo_id=None)

        filters = RecallFilters(repo_id="repo-a")
        result = self._apply_filters([m1, m2, m3], filters)
        assert len(result) == 1
        assert result[0].repo_id == "repo-a"

    def test_filter_by_task_and_agent(self):
        m1 = self._make_memory(task_id="t1", agent_id="a1")
        m2 = self._make_memory(task_id="t1", agent_id="a2")
        m3 = self._make_memory(task_id="t2", agent_id="a1")

        filters = RecallFilters(task_id="t1", agent_id="a1")
        result = self._apply_filters([m1, m2, m3], filters)
        assert len(result) == 1
        assert result[0].task_id == "t1"
        assert result[0].agent_id == "a1"

    def test_no_scope_filter_returns_all(self):
        m1 = self._make_memory(repo_id="repo-a")
        m2 = self._make_memory(repo_id="repo-b")

        filters = RecallFilters()
        result = self._apply_filters([m1, m2], filters)
        assert len(result) == 2
