"""Tests for the Claude Code hooks handler module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mnemotree.hooks.handler import (
    _handle_post_compact,
    _handle_post_tool_use_bash,
    _handle_post_tool_use_edit,
    _handle_session_start,
    _handle_stop,
    _handle_subagent_stop,
    _handle_task_completed,
    _handle_user_prompt_submit,
    handle_hook_event,
)


@pytest.fixture
def _mock_remember():
    with patch("mnemotree.hooks.handler._remember", new_callable=AsyncMock) as mock:
        mock.return_value = {"memory_id": "test-id"}
        yield mock


@pytest.fixture
def _mock_recall():
    with patch("mnemotree.hooks.handler._recall", new_callable=AsyncMock) as mock:
        mock.return_value = [
            {"snippet": "Previous context about auth", "memory_type": "semantic"},
            {"snippet": "Edited src/auth.py", "memory_type": "episodic"},
        ]
        yield mock


@pytest.fixture
def _mock_consolidate():
    with patch("mnemotree.hooks.handler._consolidate", new_callable=AsyncMock) as mock:
        mock.return_value = {"consolidated": 3}
        yield mock


@pytest.fixture
def _mock_repo():
    with (
        patch("mnemotree.hooks.handler._get_repo_id", return_value="/test/repo"),
        patch("mnemotree.hooks.handler._get_branch", return_value="feature/test"),
    ):
        yield


# ---------------------------------------------------------------------------
# SessionStart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_start_injects_context(_mock_recall, _mock_repo):
    data = {
        "hook_event_name": "SessionStart",
        "session_id": "sess-123",
        "source": "startup",
    }
    result = await _handle_session_start(data)

    assert "hookSpecificOutput" in result
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "<mnemotree-context" in ctx
    assert "Previous context about auth" in ctx


@pytest.mark.asyncio
async def test_session_start_no_memories(_mock_repo):
    with patch("mnemotree.hooks.handler._recall", new_callable=AsyncMock, return_value=[]):
        data = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-123",
            "source": "startup",
        }
        result = await _handle_session_start(data)
        assert result == {}


# ---------------------------------------------------------------------------
# PostToolUse: Edit/Write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_tool_use_edit_captures_memory(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "/test/repo/src/auth.py",
            "old_string": "def login():",
            "new_string": "def login(username: str):",
        },
        "session_id": "sess-123",
    }
    result = await _handle_post_tool_use_edit(data)

    assert result == {}
    _mock_remember.assert_awaited_once()
    call_kwargs = _mock_remember.call_args
    assert "auth.py" in call_kwargs.kwargs.get(
        "content", call_kwargs.args[0] if call_kwargs.args else ""
    )


@pytest.mark.asyncio
async def test_post_tool_use_write_captures_memory(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": "/test/repo/src/new_file.py"},
        "session_id": "sess-123",
    }
    result = await _handle_post_tool_use_edit(data)

    assert result == {}
    _mock_remember.assert_awaited_once()
    content = _mock_remember.call_args.kwargs.get("content", "")
    assert "Created/wrote" in content


# ---------------------------------------------------------------------------
# PostToolUse: Bash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_tool_use_bash_captures_significant(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "pytest tests/ -v"},
        "tool_response": "5 passed, 0 failed",
        "session_id": "sess-123",
    }
    result = await _handle_post_tool_use_bash(data)

    assert result == {}
    _mock_remember.assert_awaited_once()
    content = _mock_remember.call_args.kwargs.get("content", "")
    assert "pytest" in content


@pytest.mark.asyncio
async def test_post_tool_use_bash_skips_trivial(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "session_id": "sess-123",
    }
    result = await _handle_post_tool_use_bash(data)

    assert result == {}
    _mock_remember.assert_not_awaited()


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_creates_summary(_mock_remember, _mock_consolidate, _mock_repo):
    data = {
        "hook_event_name": "Stop",
        "session_id": "sess-123",
        "last_assistant_message": "I've finished implementing the auth module.",
    }
    result = await _handle_stop(data)

    assert result == {}
    _mock_remember.assert_awaited_once()
    _mock_consolidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_skips_when_active(_mock_remember, _mock_consolidate, _mock_repo):
    data = {
        "hook_event_name": "Stop",
        "session_id": "sess-123",
        "stop_hook_active": True,
    }
    result = await _handle_stop(data)

    assert result == {}
    _mock_remember.assert_not_awaited()


# ---------------------------------------------------------------------------
# PostCompact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_compact_saves_summary(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostCompact",
        "session_id": "sess-123",
        "compact_summary": "Working on auth refactor. Changed login flow.",
        "trigger": "auto",
    }
    result = await _handle_post_compact(data)

    assert result == {}
    _mock_remember.assert_awaited_once()
    content = _mock_remember.call_args.kwargs.get("content", "")
    assert "Compacted context" in content


@pytest.mark.asyncio
async def test_post_compact_skips_empty(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostCompact",
        "session_id": "sess-123",
        "compact_summary": "",
    }
    result = await _handle_post_compact(data)

    assert result == {}
    _mock_remember.assert_not_awaited()


# ---------------------------------------------------------------------------
# SubagentStop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_stop_captures_result(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "SubagentStop",
        "session_id": "sess-123",
        "agent_id": "agent-456",
        "agent_type": "Explore",
        "last_assistant_message": "Found 3 files matching the pattern.",
    }
    result = await _handle_subagent_stop(data)

    assert result == {}
    _mock_remember.assert_awaited_once()
    content = _mock_remember.call_args.kwargs.get("content", "")
    assert "Explore" in content


# ---------------------------------------------------------------------------
# TaskCompleted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_completed_captures(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "TaskCompleted",
        "session_id": "sess-123",
        "task_id": "task-789",
        "task_subject": "Implement login endpoint",
    }
    result = await _handle_task_completed(data)

    assert result == {}
    _mock_remember.assert_awaited_once()


# ---------------------------------------------------------------------------
# UserPromptSubmit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_prompt_submit_injects_context(_mock_recall, _mock_repo):
    data = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Fix the authentication bug in the login handler",
    }
    result = await _handle_user_prompt_submit(data)

    assert "additionalContext" in result
    ctx = result["additionalContext"]
    assert "<mnemotree-recall>" in ctx
    assert "auth" in ctx


@pytest.mark.asyncio
async def test_user_prompt_submit_skips_short(_mock_recall, _mock_repo):
    data = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "hi",
    }
    result = await _handle_user_prompt_submit(data)
    assert result == {}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_routes_correctly(_mock_remember, _mock_repo):
    data = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": "/test/repo/file.py", "old_string": "a", "new_string": "b"},
        "session_id": "sess-123",
    }
    await handle_hook_event(data)
    _mock_remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_returns_empty_for_unknown():
    result = await handle_hook_event({"hook_event_name": "UnknownEvent"})
    assert result == {}
