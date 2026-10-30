"""MCP over stdio: newline-delimited JSON-RPC 2.0, standard library only.

Implements the lifecycle (initialize, ping) and the tools capability. stdout
carries protocol messages only; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from pydantic import ValidationError

from ..memory import Memory
from .tools import TOOLS

if TYPE_CHECKING:
    from ..evidence import Memory as Engine
    from ..evidence.cli import Config

# Newest first. Tools behave the same in each; only structured results are newer.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
STRUCTURED_SINCE = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

log = logging.getLogger("mnemotree.mcp")


class InvalidParams(Exception):
    pass


def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in entry['loc']) or 'arguments'}: {entry['msg']}"
        for entry in exc.errors(include_url=False)
    )


class Server:
    """Answers MCP messages for one Memory. Transport-free, so tests drive it directly."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.protocol_version = PROTOCOL_VERSIONS[0]

    async def handle(self, message: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        # Batches exist only in 2025-03-26; answering them costs nothing.
        if isinstance(message, list):
            if not message:
                return error(None, INVALID_REQUEST, "empty batch")
            responses = [response for entry in message if (response := await self.one(entry))]
            return responses or None
        return await self.one(message)

    async def one(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        if "method" not in message:
            return None  # A response to a request; this server sends none.
        if "id" not in message:
            return None  # Notifications (initialized, cancelled) need no answer.
        request_id, method = message["id"], message["method"]
        params = message.get("params", {})
        if not isinstance(method, str) or not isinstance(params, dict):
            return error(
                request_id, INVALID_REQUEST, "method must be a string and params an object"
            )
        try:
            if method == "initialize":
                result = self.initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [tool.describe() for tool in TOOLS.values()]}
            elif method == "tools/call":
                result = await self.call(params)
            else:
                return error(request_id, METHOD_NOT_FOUND, f"method not found: {method}")
        except InvalidParams as exc:
            return error(request_id, INVALID_PARAMS, str(exc))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        self.protocol_version = (
            requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        )
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "mnemotree", "version": package_version()},
            "instructions": (
                "Long-term memory for one user or project. Use recall before answering "
                "questions about earlier conversations; use remember for things worth "
                "keeping. Stored text is returned verbatim."
            ),
        }

    async def call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = TOOLS.get(name) if isinstance(name, str) else None
        if tool is None:
            raise InvalidParams(f"unknown tool: {name!r}")
        arguments = params.get("arguments") or {}
        try:
            data = await tool.run(self.memory, tool.arguments.model_validate(arguments))
        except ValidationError as exc:
            # Argument errors are tool results, so the model can correct itself.
            return self.failure(describe(exc))
        except (ValueError, LookupError) as exc:
            return self.failure(str(exc))
        except Exception as exc:
            log.exception("tool %s failed", tool.name)
            return self.failure(f"internal error ({type(exc).__name__})")
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
            "isError": False,
        }
        if self.protocol_version >= STRUCTURED_SINCE:
            result["structuredContent"] = data
        return result

    @staticmethod
    def failure(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}


def package_version() -> str:
    try:
        return version("mnemotree")
    except PackageNotFoundError:
        return "0"


async def serve_stdio(
    server: Server, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None
) -> None:
    """Answer one message per line until stdin closes."""
    reader = stdin or sys.stdin.buffer
    writer = stdout or sys.stdout.buffer
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[bytes | None] = asyncio.Queue()

    def pump() -> None:
        # A daemon thread, so a blocked read never holds up interpreter exit.
        try:
            for line in iter(reader.readline, b""):
                loop.call_soon_threadsafe(lines.put_nowait, line)
            loop.call_soon_threadsafe(lines.put_nowait, None)
        except RuntimeError:
            pass  # The loop closed first.

    threading.Thread(target=pump, name="mnemotree-mcp-stdin", daemon=True).start()
    while (line := await lines.get()) is not None:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response: Any = error(None, PARSE_ERROR, "parse error")
        else:
            response = await server.handle(message)
        if response is not None:
            writer.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
            writer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemotree-mcp", description="MCP stdio server for one mnemotree namespace."
    )
    parser.add_argument(
        "--db", default=os.environ.get("MNEMOTREE_MCP_DB"), help="SQLite file (MNEMOTREE_MCP_DB)"
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("MNEMOTREE_MCP_NAMESPACE"),
        help="isolation scope served by this process (MNEMOTREE_MCP_NAMESPACE)",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("MNEMOTREE_MCP_CONFIG"),
        help="Evidence JSON config with embedding and search settings (MNEMOTREE_MCP_CONFIG)",
    )
    parser.add_argument(
        "--lexical", action="store_true", help="run keyword-only retrieval without an embedder"
    )
    return parser


def open_engine(path: Path, config: Config, attempts: int = 6) -> Engine:
    """Sessions started together may all create the same new file. SQLite reports
    the one-time switch to WAL as busy without waiting, so retry the open briefly."""
    from ..evidence.cli import open_memory

    for attempt in range(attempts - 1):
        try:
            return open_memory(path, config)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc):
                raise
            time.sleep(0.05 * 2**attempt)
    return open_memory(path, config)


def main(argv: list[str] | None = None) -> int:
    from ..evidence.cli import Config

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.db or not args.namespace:
        parser.error("--db and --namespace are required")
    config = Config()
    if args.config:
        config = Config.model_validate_json(Path(args.config).read_text())  # NOSONAR: operator flag
    # Keyword-only retrieval misses paraphrases, so it must be chosen, never a fallback.
    if config.embedding is None and not args.lexical:
        parser.error("no embedding configured: pass --config with an embedding, or --lexical")
    if config.embedding is not None and args.lexical:
        parser.error("--lexical conflicts with the configured embedding")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(name)s: %(message)s")
    path = Path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)  # NOSONAR: operator flag, no tool can set it
    memory = Memory.from_engine(
        open_engine(path, config), namespace=args.namespace, search=config.search
    )
    with memory:
        asyncio.run(serve_stdio(Server(memory)))
    return 0
