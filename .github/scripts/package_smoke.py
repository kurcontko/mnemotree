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


def check_location() -> None:
    import mnemotree

    origin = Path(mnemotree.__file__).resolve()
    assert "site-packages" in origin.parts, f"mnemotree imported from a checkout: {origin}"


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


def check_serve(*, bare: bool) -> None:
    for name in SERVE_MODULES:
        present = importlib.util.find_spec(name) is not None
        assert present != bare, (
            f"{name} present={present} in a {'bare' if bare else 'serve'} install"
        )
    if not bare:
        # Importing the app module is where a broken import or route shows up.
        from mnemotree.evidence import server

        assert server is not None


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
    check_cli(version)
    check_evidence_cli()
    check_serve(bare=bare)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(memory_roundtrip(Path(tmp)))
        check_mcp(Path(tmp))
        asyncio.run(evidence_roundtrip(Path(tmp)))
    print(f"mnemotree {version} ({'bare' if bare else 'serve'}): ok")


if __name__ == "__main__":
    main()
