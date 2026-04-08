#!/usr/bin/env bash
# mnemotree-capture.sh — Claude Code hook: capture conversation context as memory
#
# DEPRECATED: Use .claude/hooks/mnemotree-hook.sh instead, which handles all
# hook events via the unified mnemotree.hooks.handler module.
#
# Usage in .claude/settings.json:
#   {
#     "hooks": {
#       "PostToolUse": [
#         { "matcher": "Write|Edit", "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/mnemotree-hook.sh", "async": true}] }
#       ]
#     }
#   }
#
# This hook runs after each Write/Edit tool use and stores a brief memory
# about the file change. Requires: mnemotree installed.

set -euo pipefail

# Delegate to the unified hook handler
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
