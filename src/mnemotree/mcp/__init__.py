"""MCP server for one namespace of mnemotree memory (stdio, standard library only)."""

from .server import Server, serve_stdio
from .tools import TOOLS

__all__ = ["Server", "TOOLS", "serve_stdio"]
