"""Exercise an installed distribution from outside its checkout, without an LLM.

Run with ``--bare`` against a plain ``pip install mnemotree``, or ``--serve``
against ``mnemotree[serve]``. The MCP stdio server needs no extra, so both run
it as ``mnemotree mcp``. Run with ``--mcp`` against ``mnemotree-mcp``, the
zero-configuration server package. CI's test lanes import the package from
``src/`` and never install the wheels, so this is the only place their file
lists, metadata, and console-script entry points are checked.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The serve extra's dependencies. A bare install that pulls them in has lost
# the lean default.
SERVE_MODULES = ("fastapi", "uvicorn")
# The embeddings and lite extras' dependencies, which neither variant installs.
# If one starts arriving transitively, the lazy-import paths in
# mnemotree stop being exercised anywhere, so fail loudly instead.
# fastembed comes only with mnemotree-mcp.
OPTIONAL_MODULES = (
    "httpx",
    "tenacity",
    "sentence_transformers",
    "torch",
    "transformers",
    "fastembed",
    "onnxruntime",
)
# The point of fastembed: a local model without PyTorch.
TORCH_MODULES = ("sentence_transformers", "torch", "transformers")


def check_location() -> None:
    import mnemotree

    origin = Path(mnemotree.__file__).resolve()
    assert "site-packages" in origin.parts, f"mnemotree imported from a checkout: {origin}"
    assert origin.with_name("py.typed").is_file(), "installed package lacks its typing marker"


def check_cli(version: str) -> None:
    cli = script("mnemotree")
    out = subprocess.check_output([str(cli), "--version"], text=True)
    assert out.strip() == version, (out, version)


def check_evidence_cli() -> None:
    cli = script("mnemotree-evidence")
    subprocess.check_output([str(cli), "--help"], text=True)


def check_mcp(workdir: Path, command: list[str]) -> None:
    """A lexical session, so no model is downloaded; HOME is the scratch directory."""
    import json

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
    env = {key: value for key, value in os.environ.items() if not key.startswith("MNEMOTREE_")}
    env["HOME"] = str(workdir)
    out = subprocess.run(
        [*command, "--db", str(workdir / "mcp.sqlite"), "--namespace", "smoke", "--lexical"],
        input="".join(json.dumps(line) + "\n" for line in lines),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    ).stdout
    replies = [json.loads(line) for line in out.splitlines()]
    assert [reply["id"] for reply in replies] == [0, 1, 2], replies
    assert replies[0]["result"]["serverInfo"]["name"] == "mnemotree", replies[0]
    assert replies[2]["result"]["isError"] is False, replies[2]
    assert not (workdir / ".mnemotree").exists(), "a lexical session touched the model cache"


def script(name: str) -> Path:
    return Path(sys.executable).parent / name


def check_mcp_package(version: str) -> None:
    """mnemotree-mcp: same version as mnemotree, pinned to it, owning the script."""
    import re

    import mnemotree_mcp

    origin = Path(mnemotree_mcp.__file__).resolve()
    assert "site-packages" in origin.parts, f"mnemotree_mcp imported from a checkout: {origin}"
    assert importlib.metadata.version("mnemotree-mcp") == version
    # The wheel just built, not the release of the same version on PyPI.
    source = importlib.metadata.distribution("mnemotree").read_text("direct_url.json") or ""
    assert '"file://' in source, f"mnemotree was not installed from the local wheel: {source}"
    pins = [
        requirement
        for requirement in importlib.metadata.requires("mnemotree-mcp") or []
        if re.split(r"[\s\[=<>!~;]", requirement, maxsplit=1)[0] == "mnemotree"
    ]
    assert pins == [f"mnemotree=={version}"], pins
    (entry,) = importlib.metadata.entry_points(group="console_scripts", name="mnemotree-mcp")
    assert entry.dist is not None and entry.dist.name == "mnemotree-mcp", entry
    present = [name for name in TORCH_MODULES if importlib.util.find_spec(name) is not None]
    assert not present, f"mnemotree-mcp pulled in {present}"
    out = subprocess.check_output([str(script("mnemotree-mcp")), "--help"], text=True)
    assert "granite-embedding-small-english-r2" in out, out


def check_default_model() -> None:
    """The installed fastembed knows the pinned model, so its files resolve offline."""
    from mnemotree.providers.embeddings import DEFAULT_LOCAL_EMBEDDING, FastEmbedEmbedder

    settings = dict(DEFAULT_LOCAL_EMBEDDING)
    settings.pop("kind")
    files = FastEmbedEmbedder(**settings).files()
    assert files[-2:] == ["onnx/model.onnx", "onnx/model.onnx_data"], files


def check_optional_absent() -> None:
    present = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None]
    assert not present, f"Unexpected optional dependency: {present}"


def check_serve(*, bare: bool) -> None:
    for name in SERVE_MODULES:
        present = importlib.util.find_spec(name) is not None
        assert present != bare, (
            f"{name} present={present} in a {'bare' if bare else 'serve'} install"
        )


async def asgi_call(
    app: Any, method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
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

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
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
    from mnemotree.service.app import create_app

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


def check_legacy_paths(*, bare: bool) -> None:
    """The 1.0 mnemotree.evidence.<module> paths hand out the moved modules' objects."""
    moved = {
        "engine": "mnemotree.engine",
        "models": "mnemotree.core.models",
        "store": "mnemotree.storage.store",
        "atoms": "mnemotree.ingest.atoms",
        "providers": "mnemotree.providers.embeddings",
        "cli": "mnemotree.cli.evidence",
    }
    if not bare:
        moved["server"] = "mnemotree.service.app"
    for old, new in moved.items():
        module = importlib.import_module(f"mnemotree.evidence.{old}")
        target = importlib.import_module(new)
        # 1.0's secret exits on a missing credential; the moved one raises.
        kept = {"secret"} if old == "cli" else set()
        public = [
            name
            for name in vars(module)
            if not name.startswith("_") and name not in kept and hasattr(target, name)
        ]
        assert public, (old, new)
        assert all(getattr(module, name) is getattr(target, name) for name in public), (old, new)


def main() -> None:
    variant = sys.argv[1].removeprefix("--") if len(sys.argv) > 1 else "bare"
    assert variant in ("bare", "serve", "mcp"), variant
    version = importlib.metadata.version("mnemotree")

    check_location()
    check_cli(version)
    check_evidence_cli()
    if variant == "mcp":
        check_mcp_package(version)
        check_default_model()
        with tempfile.TemporaryDirectory() as tmp:
            check_mcp(Path(tmp), [str(script("mnemotree-mcp"))])
        print(f"mnemotree-mcp {version}: ok")
        return
    check_optional_absent()
    check_serve(bare=variant == "bare")
    check_legacy_paths(bare=variant == "bare")
    # The mnemotree-mcp distribution owns that script; two owning it would clash.
    assert not script("mnemotree-mcp").exists(), "mnemotree installs a mnemotree-mcp script"
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(memory_roundtrip(Path(tmp)))
        check_mcp(Path(tmp), [str(script("mnemotree")), "mcp"])
        asyncio.run(evidence_roundtrip(Path(tmp)))
        if variant == "serve":
            asyncio.run(service_roundtrip(Path(tmp)))
    print(f"mnemotree {version} ({variant}): ok")


if __name__ == "__main__":
    main()
