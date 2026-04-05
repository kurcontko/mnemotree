#!/usr/bin/env bash
# mnemotree-augment.sh — Claude Code hook: augment prompts with relevant memories
#
# Usage in .claude/settings.json:
#   {
#     "hooks": {
#       "PreToolUse": [
#         { "matcher": ".*", "command": "bash examples/cli-hooks/mnemotree-augment.sh" }
#       ]
#     }
#   }
#
# This hook runs before each tool use and injects relevant memories
# into the context as system reminders. Helps Claude maintain long-term
# project awareness across sessions.

set -euo pipefail

# Read the tool input from stdin
INPUT=$(cat)

# Extract the query/content from the input
QUERY=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
# Use tool input content as query, or the tool name as fallback
content = data.get('tool_input', {}).get('content', '')
if not content:
    content = data.get('tool_input', {}).get('command', '')
if not content:
    content = data.get('tool_name', '')
# Truncate for search
print(content[:200])
" 2>/dev/null || echo "")

if [ -z "$QUERY" ]; then
    exit 0
fi

# Recall relevant memories and print as context
python3 -c "
import asyncio
from mnemotree.mcp.server import recall

async def main():
    query = '''$QUERY'''
    if not query.strip():
        return
    memories = await recall(query=query, limit=3, compact=True)
    if memories:
        print('<mnemotree-context>')
        for mem in memories:
            print(f\"  [{mem.get('memory_type', '?')}] {mem.get('snippet', '')}\")
        print('</mnemotree-context>')

asyncio.run(main())
" 2>/dev/null || true
