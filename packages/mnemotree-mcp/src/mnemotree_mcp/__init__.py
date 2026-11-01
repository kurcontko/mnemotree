"""Mnemotree's MCP stdio server, defaulting to a local embedding model."""

from __future__ import annotations

from mnemotree.mcp.server import main as serve
from mnemotree.providers.embeddings import DEFAULT_LOCAL_EMBEDDING


def main(argv: list[str] | None = None) -> int:
    return serve(argv, default_embedding=DEFAULT_LOCAL_EMBEDDING)
