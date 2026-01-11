from __future__ import annotations

import json
from typing import Any

JSON_RPC_VERSION = "2.0"
ENCODING = "utf-8"


class LspProtocolError(Exception):
    pass


def make_request(method: str, params: Any, request_id: int | str) -> dict[str, Any]:
    return {
        "jsonrpc": JSON_RPC_VERSION,
        "method": method,
        "params": params,
        "id": request_id,
    }


def make_notification(method: str, params: Any) -> dict[str, Any]:
    return {
        "jsonrpc": JSON_RPC_VERSION,
        "method": method,
        "params": params,
    }


def encode_message(content: dict[str, Any]) -> bytes:
    """Encodes a JSON content into an LSP message with headers."""
    content_json = json.dumps(content, separators=(",", ":")).encode(ENCODING)
    length = len(content_json)
    header = f"Content-Length: {length}\r\n\r\n".encode(ENCODING)
    return header + content_json


def decode_header(header: bytes) -> int:
    """Parses Content-Length from the header."""
    header_str = header.decode(ENCODING)
    for line in header_str.split("\r\n"):
        if line.startswith("Content-Length:"):
            return int(line.split(":", 1)[1].strip())
    raise LspProtocolError(f"Invalid header: {header_str}")
