"""Claude Code hooks integration for mnemotree.

Captures memories, iterations, and context from all Claude Code instances
via the hooks lifecycle (SessionStart, PostToolUse, Stop, PostCompact, etc.).

Designed for multi-instance awareness using scoped memory (repo_id, session_id).
"""

from mnemotree.hooks.handler import handle_hook_event

__all__ = ["handle_hook_event"]
