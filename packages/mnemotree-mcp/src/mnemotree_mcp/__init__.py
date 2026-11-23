"""Mnemotree's MCP stdio server, defaulting to a local embedding model."""

from __future__ import annotations

import sys

from mnemotree.mcp.server import main as serve
from mnemotree.providers.embeddings import DEFAULT_LOCAL_EMBEDDING


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # `mnemotree-mcp hook` handles a Claude Code hook event with the same local model,
    # so the plugin's hooks and its MCP server read and write one database.
    if argv[:1] == ["hook"]:
        from mnemotree.claude_code import main as hook

        return hook(
            argv[1:],
            default_embedding=DEFAULT_LOCAL_EMBEDDING,
            command=(sys.executable, "-m", "mnemotree_mcp", "hook"),
        )
    return serve(argv, default_embedding=DEFAULT_LOCAL_EMBEDDING)
