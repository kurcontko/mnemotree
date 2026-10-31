"""Exercise an installed distribution from outside its checkout, without an LLM.

Run with ``--bare`` against a plain ``pip install mnemotree``, or ``--serve``
against ``mnemotree[serve]``. The MCP stdio server needs no extra, so both run
it. CI's test lanes import the package from ``src/`` and never install the
wheel, so this is the only place the wheel's file list, metadata, and
console-script entry points are checked.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

# The serve extra's dependencies. A bare install that pulls them in has lost
# the lean default.
SERVE_MODULES = ("fastapi", "uvicorn")
# The embeddings and lite extras' dependencies, which neither variant installs.
# If one starts arriving transitively, the lazy-import paths in
# mnemotree.evidence stop being exercised anywhere, so fail loudly instead.
OPTIONAL_MODULES = ("httpx", "tenacity", "sentence_transformers", "torch", "transformers")


def check_location() -> None:
    import mnemotree

    origin = Path(mnemotree.__file__).resolve()
    assert "site-packages" in origin.parts, f"mnemotree imported from a checkout: {origin}"
    assert origin.with_name("py.typed").is_file(), "installed package lacks its typing marker"


def check_cli(version: str) -> None:
    cli = Path(sys.executable).parent / "mnemotree"
    out = subprocess.check_output([str(cli), "--version"], text=True)
    assert out.strip() == version, (out, version)


def check_evidence_cli() -> None:
    cli = Path(sys.executable).parent / "mnemotree-evidence"
    subprocess.check_output([str(cli), "--help"], text=True)


def check_mcp(workdir: Path) -> None:
    import json

    cli = Path(sys.executable).parent / "mnemotree-mcp"
    lines = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "remember", "arguments": {"content": "MCP smoke memory."}},
        },
    ]
    out = subprocess.run(
        [str(cli), "--db", str(workdir / "mcp.sqlite"), "--namespace", "smoke", "--lexical"],
        input="".join(json.dumps(line) + "\n" for line in lines),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    replies = [json.loads(line) for line in out.splitlines()]
    assert [reply["id"] for reply in replies] == [0, 1, 2], replies
    assert replies[2]["result"]["isError"] is False, replies[2]


def check_optional_absent() -> None:
    present = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None]
    assert not present, f"Unexpected optional dependency: {present}"


def check_serve(*, bare: bool) -> None:
    for name in SERVE_MODULES:
        present = importlib.util.find_spec(name) is not None
        assert present != bare, (
            f"{name} present={present} in a {'bare' if bare else 'serve'} install"
        )


async def asgi_call(app, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """One request through the app with no HTTP client: the serve extra has none."""
    import json

    payload = json.dumps(body).encode() if body is not None else b""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer smoke-key"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 8000),
    }
    sent = False
    status = 0
    chunks: list[bytes] = []

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    data = b"".join(chunks)
    return status, json.loads(data) if data else {}


async def service_roundtrip(workdir: Path) -> None:
    """Build the AML app and send Add and Search through its routes and middleware."""
    from mnemotree.evidence import Memory
    from mnemotree.evidence.server import create_app

    with Memory(workdir / "service.sqlite") as memory:
        app = create_app(memory, api_key="smoke-key")
        assert await asgi_call(app, "GET", "/health") == (200, {"status": "ok"})
        status, body = await asgi_call(
            app,
            "POST",
            "/v1/memories/add",
            {
                "request_id": "smoke-1",
                "user_id": "smoke",
                "session_id": "smoke",
                "messages": [
                    {"role": "user", "content": "The package smoke test reached the service."}
                ],
            },
        )
        assert status == 200, (status, body)
        status, body = await asgi_call(
            app,
            "POST",
            "/v1/memories/search",
            {"user_id": "smoke", "query": "package smoke service", "top_k": 5},
        )
        assert status == 200, (status, body)
        assert [item["content"] for item in body["data"]] == [
            "user: The package smoke test reached the service."
        ], body


async def memory_roundtrip(workdir: Path) -> None:
    from mnemotree import Memory

    with Memory(workdir / "memory.sqlite", namespace="smoke") as memory:
        item = await memory.remember("The package smoke test stored this memory.")
        recall = await memory.recall("package smoke memory")
        assert [found.id for found in recall.items] == [item.id], recall.items


async def evidence_roundtrip(workdir: Path) -> None:
    from mnemotree.evidence import Memory, Message

    with Memory(workdir / "evidence.sqlite") as memory:
        await memory.add(
            "smoke",
            [Message(content="The package smoke test stored this evidence.")],
            request_id="smoke-1",
            session_id="smoke",
        )
        result = await memory.search("smoke", "package smoke evidence")
        assert [item.text for item in result.evidence] == [
            "The package smoke test stored this evidence."
        ], result.evidence


def main() -> None:
    bare = "--bare" in sys.argv
    version = importlib.metadata.version("mnemotree")

    check_location()
    check_optional_absent()
    check_cli(version)
    check_evidence_cli()
    check_serve(bare=bare)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(memory_roundtrip(Path(tmp)))
        check_mcp(Path(tmp))
        asyncio.run(evidence_roundtrip(Path(tmp)))
        if not bare:
            asyncio.run(service_roundtrip(Path(tmp)))
    print(f"mnemotree {version} ({'bare' if bare else 'serve'}): ok")


if __name__ == "__main__":
    main()
