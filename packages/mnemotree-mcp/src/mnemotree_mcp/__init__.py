"""Mnemotree's MCP stdio server, defaulting to a local embedding model."""

from __future__ import annotations

from mnemotree.evidence.providers import DEFAULT_LOCAL_EMBEDDING
from mnemotree.mcp.server import main as serve


def main(argv: list[str] | None = None) -> int:
    return serve(argv, default_embedding=DEFAULT_LOCAL_EMBEDDING)
