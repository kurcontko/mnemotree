#!/usr/bin/env bash
# mnemotree-reflect.sh — Claude Code hook: end-of-session reflection
#
# DEPRECATED: Use .claude/hooks/mnemotree-hook.sh instead, which handles
# Stop events via the unified mnemotree.hooks.handler module.
# The Stop hook now also includes a prompt-based continuation checker.
#
# Usage in .claude/settings.json:
#   {
#     "hooks": {
#       "Stop": [
#         { "hooks": [
#           {"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "async": true},
#           {"type": "prompt", "prompt": "Check if all tasks are complete..."}
#         ]}
#       ]
#     }
#   }
#
# This hook runs when a Claude Code session ends. It triggers memory
# consolidation and stores a session summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
exec "$PYTHON" -m mnemotree.hooks.handler
