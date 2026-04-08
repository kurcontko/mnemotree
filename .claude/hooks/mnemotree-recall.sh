#!/usr/bin/env bash
# mnemotree-recall.sh — Lightweight recall hook for UserPromptSubmit
#
# Runs on every user prompt to inject relevant memories as context.
# Kept separate for performance: this is the hot path and should be fast.
# Falls back gracefully if mnemotree is not installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    exit 0
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Read stdin, extract prompt, recall relevant memories
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | "$PYTHON" -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('prompt', '')[:300])
" 2>/dev/null || echo "")

if [ -z "$PROMPT" ] || [ ${#PROMPT} -lt 10 ]; then
    exit 0
fi

# Recall and output context
"$PYTHON" -c "
import asyncio, json, sys, os
sys.path.insert(0, os.path.join('$PROJECT_DIR', 'src'))

async def main():
    try:
        from mnemotree.mcp.server import recall
        query = '''$PROMPT'''
        results = await recall(query=query, limit=3, compact=True)
        if results:
            lines = ['<mnemotree-recall>']
            for mem in (results if isinstance(results, list) else []):
                snippet = mem.get('snippet', mem.get('content', ''))[:250]
                mtype = mem.get('memory_type', '?')
                lines.append(f'  [{mtype}] {snippet}')
            lines.append('</mnemotree-recall>')
            print('\n'.join(lines))
    except Exception:
        pass

asyncio.run(main())
" 2>/dev/null || true
