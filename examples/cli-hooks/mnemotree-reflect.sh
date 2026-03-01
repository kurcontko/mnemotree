#!/usr/bin/env bash
# mnemotree-reflect.sh — Claude Code hook: end-of-session reflection
#
# Usage in .claude/settings.json:
#   {
#     "hooks": {
#       "Stop": [
#         { "command": "bash examples/cli-hooks/mnemotree-reflect.sh" }
#       ]
#     }
#   }
#
# This hook runs when a Claude Code session ends. It triggers memory
# consolidation (if enough memories have accumulated) and stores a
# session summary. Inspired by the "sleep consolidation" pattern from
# RAPTOR/MAPLE research.

set -euo pipefail

python3 -c "
import asyncio
from datetime import datetime, timezone

from mnemotree.mcp.server import recall, remember, reflect

async def main():
    # 1. Gather recent session memories
    recent = await recall(
        query='session activity code changes',
        limit=10,
        compact=True,
    )

    if not recent:
        return

    # 2. Create a session summary
    snippets = [m.get('snippet', '') for m in recent if m.get('snippet')]
    if snippets:
        summary = 'Session summary: ' + '; '.join(snippets[:5])
        await remember(
            content=summary,
            memory_type='semantic',
            importance=0.7,
            tags=['session-summary', 'consolidation'],
            context={
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory_count': len(recent),
            },
        )

    # 3. Run reflection on high-importance memories
    try:
        reflection = await reflect(min_importance=0.7)
        if reflection and reflection.get('summary'):
            print(f\"Reflection: {reflection['summary'][:200]}\")
    except Exception:
        pass  # Reflection requires LLM, skip in lite mode

asyncio.run(main())
" 2>/dev/null || true
