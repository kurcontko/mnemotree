#!/usr/bin/env bash
# mnemotree-capture.sh — Claude Code hook: capture conversation context as memory
#
# Usage in .claude/settings.json:
#   {
#     "hooks": {
#       "PostToolUse": [
#         { "matcher": "Write|Edit", "command": "bash examples/cli-hooks/mnemotree-capture.sh" }
#       ]
#     }
#   }
#
# This hook runs after each Write/Edit tool use and stores a brief memory
# about the file change. Requires: mnemotree[mcp_server] or the Python package.

set -euo pipefail

# Read the tool result from stdin (Claude Code pipes JSON)
INPUT=$(cat)

# Extract file path from the tool result
FILE_PATH=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
# Adjust based on actual hook payload structure
path = data.get('tool_input', {}).get('file_path', '')
if not path:
    path = data.get('file_path', 'unknown')
print(path)
" 2>/dev/null || echo "unknown")

# Skip if no meaningful file path
if [ "$FILE_PATH" = "unknown" ] || [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Store memory about this edit
python3 -c "
import asyncio
from mnemotree.mcp.server import remember

async def main():
    await remember(
        content='Edited file: $FILE_PATH',
        memory_type='episodic',
        tags=['code-edit', 'claude-code'],
        context={'file': '$FILE_PATH', 'hook': 'PostToolUse'},
    )

asyncio.run(main())
" 2>/dev/null || true
