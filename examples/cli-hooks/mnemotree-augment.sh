#!/usr/bin/env bash
# mnemotree-augment.sh — Claude Code hook: augment prompts with relevant memories
#
# DEPRECATED: Use .claude/hooks/mnemotree-recall.sh instead, which handles
# UserPromptSubmit recall via the unified mnemotree.hooks.handler module.
#
# Usage in .claude/settings.json:
#   {
#     "hooks": {
#       "UserPromptSubmit": [
#         { "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-recall.sh"}] }
#       ]
#     }
#   }
#
# This hook runs on each user prompt and injects relevant memories
# into the context. Helps Claude maintain long-term project awareness.

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
