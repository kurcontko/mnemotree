#!/usr/bin/env bash
# mnemotree-hook.sh — Universal Claude Code hook dispatcher for mnemotree
#
# This single script handles ALL hook events by delegating to the Python
# handler module. Claude Code pipes event JSON to stdin; we forward it
# to `python -m mnemotree.hooks.handler` and relay the output.
#
# Usage: Configured in .claude/settings.json or ~/.claude/settings.json
# All hook events route through this script.

set -euo pipefail

# Find the project root (where this script lives under .claude/hooks/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use project's Python if available (uv/venv), else system Python
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Ensure mnemotree is importable
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Forward stdin to the Python handler; relay stdout/stderr
exec "$PYTHON" -m mnemotree.hooks.handler
