"""Core hook event handler for Claude Code ↔ mnemotree integration.

Each hook event is dispatched to a handler function that decides what to
remember, recall, or consolidate.  The handler reads JSON from stdin
(provided by Claude Code) and writes JSON or plain text to stdout
(injected back into Claude Code context).

Scoping: every memory is tagged with ``repo_id`` (from git toplevel) and
``session_id`` (from the hook payload) so that multiple Claude Code
instances writing to the same mnemotree store don't collide.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_repo_id() -> str:
    """Derive a stable repo identifier from the git toplevel path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())


def _get_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Memory operations (async, using mnemotree MCP server functions directly)
# ---------------------------------------------------------------------------


async def _remember(
    content: str,
    memory_type: str = "episodic",
    importance: float = 0.5,
    tags: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Store a memory via the MCP server's remember() function."""
    try:
        from mnemotree.mcp.server import remember

        result = await remember(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            context=context or {},
        )
        return result
    except Exception as exc:
        logger.debug("remember failed: %s", exc)
        return None


async def _recall(
    query: str,
    limit: int = 5,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recall memories via the MCP server's recall() function."""
    try:
        from mnemotree.mcp.server import recall

        results = await recall(query=query, limit=limit, compact=True)
        return results if isinstance(results, list) else []
    except Exception as exc:
        logger.debug("recall failed: %s", exc)
        return []


async def _consolidate() -> dict[str, Any] | None:
    """Trigger memory consolidation."""
    try:
        from mnemotree.mcp.server import consolidate

        return await consolidate()
    except Exception as exc:
        logger.debug("consolidate failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


async def _handle_session_start(data: dict[str, Any]) -> dict[str, Any]:
    """SessionStart: inject recalled context into the new session.

    On 'startup'/'resume': recall recent project context.
    On 'compact': re-inject critical memories that may have been lost.
    """
    source = data.get("source", "startup")
    session_id = data.get("session_id", "unknown")
    repo_id = _get_repo_id()
    branch = _get_branch()

    # Recall recent context relevant to this repo
    query = f"project context for {os.path.basename(repo_id)} branch {branch}"
    memories = await _recall(query=query, limit=5, tags=["claude-code"])

    if not memories:
        # No memories yet — just note the session start
        return {}

    # Build context injection
    lines = [f'<mnemotree-context source="{source}" session="{session_id[:8]}">']
    for mem in memories:
        snippet = mem.get("snippet", mem.get("content", ""))
        mtype = mem.get("memory_type", "?")
        lines.append(f"  [{mtype}] {_truncate(snippet, 300)}")
    lines.append("</mnemotree-context>")

    context_text = "\n".join(lines)

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context_text,
        }
    }


async def _handle_post_tool_use_edit(data: dict[str, Any]) -> dict[str, Any]:
    """PostToolUse for Write/Edit: capture file changes as episodic memories."""
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "unknown")
    repo_id = _get_repo_id()
    branch = _get_branch()

    file_path = tool_input.get("file_path", "unknown")

    # Make path relative to repo
    try:
        rel_path = os.path.relpath(file_path, repo_id)
    except ValueError:
        rel_path = file_path

    # Build a concise memory of this edit
    if tool_name == "Write":
        content = f"Created/wrote file: {rel_path}"
    else:
        old_str = _truncate(tool_input.get("old_string", ""), 100)
        new_str = _truncate(tool_input.get("new_string", ""), 100)
        content = f"Edited {rel_path}: '{old_str}' → '{new_str}'"

    await _remember(
        content=content,
        memory_type="episodic",
        importance=0.4,
        tags=["claude-code", "file-edit", branch],
        context={
            "file": rel_path,
            "tool": tool_name,
            "session_id": session_id,
            "repo_id": repo_id,
            "branch": branch,
            "timestamp": _now_iso(),
        },
    )
    return {}


async def _handle_post_tool_use_bash(data: dict[str, Any]) -> dict[str, Any]:
    """PostToolUse for Bash: capture significant commands."""
    tool_input = data.get("tool_input", {})
    tool_response = data.get("tool_response", "")
    session_id = data.get("session_id", "unknown")
    repo_id = _get_repo_id()
    branch = _get_branch()

    command = tool_input.get("command", "")

    # Skip trivial commands
    trivial_prefixes = ("ls", "cat ", "echo ", "pwd", "cd ", "head ", "tail ")
    if any(command.strip().startswith(p) for p in trivial_prefixes):
        return {}

    # Capture test results, installs, builds, git operations
    significant_keywords = (
        "test",
        "pytest",
        "npm run",
        "make",
        "build",
        "install",
        "git commit",
        "git push",
        "git merge",
        "git rebase",
        "docker",
        "deploy",
        "migrate",
    )
    is_significant = any(kw in command for kw in significant_keywords)
    if not is_significant:
        return {}

    # Truncate response for storage
    response_snippet = _truncate(str(tool_response), 300) if tool_response else ""

    content = f"Ran: `{_truncate(command, 200)}`"
    if response_snippet:
        content += f"\nResult: {response_snippet}"

    await _remember(
        content=content,
        memory_type="episodic",
        importance=0.5,
        tags=["claude-code", "command", branch],
        context={
            "command": command,
            "session_id": session_id,
            "repo_id": repo_id,
            "branch": branch,
            "timestamp": _now_iso(),
        },
    )
    return {}


async def _handle_stop(data: dict[str, Any]) -> dict[str, Any]:
    """Stop: session reflection and consolidation.

    Captures a session summary and triggers consolidation if enough
    memories have accumulated.
    """
    # Avoid infinite loop: if stop_hook_active, let Claude stop
    if data.get("stop_hook_active"):
        return {}

    session_id = data.get("session_id", "unknown")
    repo_id = _get_repo_id()
    branch = _get_branch()
    last_message = data.get("last_assistant_message", "")

    # Store session end marker with last assistant message summary
    if last_message:
        summary = _truncate(last_message, 400)
        await _remember(
            content=f"Session end summary: {summary}",
            memory_type="semantic",
            importance=0.6,
            tags=["claude-code", "session-summary", branch],
            context={
                "session_id": session_id,
                "repo_id": repo_id,
                "branch": branch,
                "timestamp": _now_iso(),
            },
        )

    # Trigger consolidation in the background (non-blocking)
    await _consolidate()

    return {}


async def _handle_post_compact(data: dict[str, Any]) -> dict[str, Any]:
    """PostCompact: save the compaction summary before context is lost."""
    session_id = data.get("session_id", "unknown")
    repo_id = _get_repo_id()
    branch = _get_branch()
    compact_summary = data.get("compact_summary", "")

    if not compact_summary:
        return {}

    await _remember(
        content=f"Compacted context: {_truncate(compact_summary, 800)}",
        memory_type="semantic",
        importance=0.7,
        tags=["claude-code", "compaction", branch],
        context={
            "session_id": session_id,
            "repo_id": repo_id,
            "branch": branch,
            "trigger": data.get("trigger", "auto"),
            "timestamp": _now_iso(),
        },
    )
    return {}


async def _handle_subagent_stop(data: dict[str, Any]) -> dict[str, Any]:
    """SubagentStop: capture subagent results for cross-instance awareness."""
    session_id = data.get("session_id", "unknown")
    agent_id = data.get("agent_id", "unknown")
    agent_type = data.get("agent_type", "unknown")
    repo_id = _get_repo_id()
    branch = _get_branch()
    last_message = data.get("last_assistant_message", "")

    if not last_message:
        return {}

    await _remember(
        content=f"Subagent [{agent_type}] result: {_truncate(last_message, 400)}",
        memory_type="episodic",
        importance=0.5,
        tags=["claude-code", "subagent", agent_type, branch],
        context={
            "agent_id": agent_id,
            "agent_type": agent_type,
            "session_id": session_id,
            "repo_id": repo_id,
            "branch": branch,
            "timestamp": _now_iso(),
        },
    )
    return {}


async def _handle_task_completed(data: dict[str, Any]) -> dict[str, Any]:
    """TaskCompleted: capture completed task info for iteration tracking."""
    session_id = data.get("session_id", "unknown")
    task_id = data.get("task_id", "")
    task_subject = data.get("task_subject", "")
    repo_id = _get_repo_id()
    branch = _get_branch()

    if not task_subject:
        return {}

    await _remember(
        content=f"Task completed: {task_subject}",
        memory_type="episodic",
        importance=0.6,
        tags=["claude-code", "task-completed", branch],
        context={
            "task_id": task_id,
            "session_id": session_id,
            "repo_id": repo_id,
            "branch": branch,
            "timestamp": _now_iso(),
        },
    )
    return {}


async def _handle_user_prompt_submit(data: dict[str, Any]) -> dict[str, Any]:
    """UserPromptSubmit: recall context relevant to the user's prompt.

    Injects relevant memories as additional context so Claude has
    cross-session awareness.
    """
    prompt = data.get("prompt", "")
    if not prompt or len(prompt) < 10:
        return {}

    branch = _get_branch()

    # Recall memories relevant to this prompt
    query = f"{prompt} branch:{branch}"
    memories = await _recall(query=_truncate(query, 300), limit=3)

    if not memories:
        return {}

    lines = ["<mnemotree-recall>"]
    for mem in memories:
        snippet = mem.get("snippet", mem.get("content", ""))
        mtype = mem.get("memory_type", "?")
        lines.append(f"  [{mtype}] {_truncate(snippet, 250)}")
    lines.append("</mnemotree-recall>")

    # Return as additionalContext for UserPromptSubmit
    return {
        "additionalContext": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "SessionStart": _handle_session_start,
    "PostToolUse:Edit": _handle_post_tool_use_edit,
    "PostToolUse:Write": _handle_post_tool_use_edit,
    "PostToolUse:Bash": _handle_post_tool_use_bash,
    "Stop": _handle_stop,
    "PostCompact": _handle_post_compact,
    "SubagentStop": _handle_subagent_stop,
    "TaskCompleted": _handle_task_completed,
    "UserPromptSubmit": _handle_user_prompt_submit,
}


async def handle_hook_event(data: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a hook event to the appropriate handler.

    Args:
        data: The JSON payload from Claude Code's hook system (stdin).

    Returns:
        A dict to be serialized as JSON to stdout. Empty dict = no action.
    """
    event_name = data.get("hook_event_name", "")
    tool_name = data.get("tool_name", "")

    # Try specific handler first (e.g., PostToolUse:Edit)
    specific_key = f"{event_name}:{tool_name}" if tool_name else ""
    handler = _HANDLERS.get(specific_key) or _HANDLERS.get(event_name)

    if handler is None:
        return {}

    try:
        return await handler(data)
    except Exception as exc:
        logger.debug("Hook handler %s failed: %s", event_name, exc)
        return {}


# ---------------------------------------------------------------------------
# CLI entry point (called by hook scripts)
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: read JSON from stdin, dispatch, write JSON to stdout."""
    import asyncio

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        print(f"Hook input parse error: {exc}", file=sys.stderr)
        sys.exit(0)  # Exit 0 = don't block

    result = asyncio.run(handle_hook_event(data))

    if result:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
