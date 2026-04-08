# Claude Code Hooks Integration

Mnemotree integrates with Claude Code's [hook lifecycle](https://code.claude.com/docs/en/hooks) to automatically capture memories, iterations, and context from **all** Claude Code instances working on your project.

## What Gets Captured

| Hook Event | What's Stored | Memory Type |
|---|---|---|
| **SessionStart** | Recalls relevant context, injects into new session | (recall only) |
| **UserPromptSubmit** | Recalls memories relevant to each prompt | (recall only) |
| **PostToolUse** (Edit/Write) | File changes with before/after snippets | episodic |
| **PostToolUse** (Bash) | Significant commands (tests, builds, deploys) | episodic |
| **Stop** | Session summary + triggers consolidation | semantic |
| **PostCompact** | Compaction summary (saves context before it's lost) | semantic |
| **SubagentStop** | Subagent results (Explore, Plan, etc.) | episodic |
| **TaskCompleted** | Completed task subjects | episodic |

## Quick Setup

### 1. Install mnemotree

```bash
pip install mnemotree
# or with uv:
uv pip install mnemotree
```

### 2. Copy the hook configuration

The hooks are pre-configured in `.claude/settings.json`. If you're adding to an existing project:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|compact",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 15 }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 10, "async": true }]
      },
      {
        "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 10, "async": true }]
      }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 30, "async": true }] }
    ],
    "PostCompact": [
      { "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 15, "async": true }] }
    ],
    "SubagentStop": [
      { "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 15, "async": true }] }
    ],
    "TaskCompleted": [
      { "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "timeout": 10, "async": true }] }
    ]
  }
}
```

### 3. Copy the hook scripts

```bash
mkdir -p .claude/hooks
cp examples/cli-hooks/mnemotree-hook.sh .claude/hooks/
chmod +x .claude/hooks/mnemotree-hook.sh
```

Or use the scripts already in `.claude/hooks/` if you cloned this repo.

## Architecture

```
Claude Code Instance 1 ──┐
Claude Code Instance 2 ──┤── hooks (stdin JSON) ──→ mnemotree-hook.sh
Claude Code Instance 3 ──┘                              │
                                                         ▼
                                                  mnemotree.hooks.handler
                                                         │
                                          ┌──────────────┼──────────────┐
                                          ▼              ▼              ▼
                                      remember()     recall()    consolidate()
                                          │              │              │
                                          └──────────────┴──────────────┘
                                                         │
                                                         ▼
                                                  Shared SQLite DB
                                              (or ChromaDB / Neo4j)
```

**Multi-instance safety**: Every memory is tagged with `repo_id`, `session_id`, and `branch`. The shared store handles concurrent writes via SQLite WAL mode.

## How It Works

1. **SessionStart**: When a new Claude Code session starts (or resumes, or compacts), the hook recalls the 5 most relevant memories for the current project/branch and injects them as `additionalContext`.

2. **UserPromptSubmit**: On every user prompt, recalls 3 relevant memories and injects them as context so Claude has cross-session awareness.

3. **PostToolUse** (async): After file edits or significant bash commands, stores an episodic memory with the change details. Runs async so it doesn't slow down the session.

4. **Stop**: When Claude finishes responding, stores a session summary and triggers memory consolidation (RAPTOR-style cluster+summarize).

5. **PostCompact**: When context is compacted, saves the summary to prevent knowledge loss.

6. **SubagentStop/TaskCompleted** (async): Captures subagent results and completed tasks for cross-instance iteration tracking.

## Customization

### Filtering Bash commands

Edit `_handle_post_tool_use_bash` in `src/mnemotree/hooks/handler.py` to adjust which commands are captured. By default, trivial commands (`ls`, `cat`, `echo`, etc.) are skipped.

### Recall limits

Adjust the `limit` parameter in `_handle_session_start` and `_handle_user_prompt_submit` to control how many memories are injected.

### agent-layer-clean compatibility

The `agent-layer-clean` branch adds scoped memory (repo_id, worktree_id, task_id, agent_id, run_id). When merged, the hooks will automatically use these scoping fields for full multi-agent isolation.

## Programmatic Usage

```python
from mnemotree.hooks import handle_hook_event

# Process a hook event manually
result = await handle_hook_event({
    "hook_event_name": "PostToolUse",
    "tool_name": "Edit",
    "tool_input": {"file_path": "src/main.py", "old_string": "...", "new_string": "..."},
    "session_id": "my-session",
})
```
