"""Exercise an installed distribution from outside its checkout, without an LLM.

Run with ``--bare`` against a plain ``pip install mnemotree``, or ``--mcp``
against ``mnemotree[mcp]``. The bare install is the evidence engine alone;
``mcp`` adds the legacy MemoryCore that the MCP server runs on. CI's test lanes import the package from ``src/``
and never install the wheel, so this is the only place the wheel's file list,
metadata, and console-script entry points are checked.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

# Optional backends a bare install must not pull in. If one starts arriving
# transitively, the soft-import paths in store/ and embeddings/ stop being
# exercised anywhere, so fail loudly instead.
OPTIONAL_MODULES = ("chromadb", "fastmcp", "sentence_transformers", "neo4j", "pymilvus")
# MemoryCore's dependencies live in the legacy extra; a bare install that
# pulls them in has lost the lean default.
LEGACY_MODULES = ("langchain", "langchain_core", "numpy", "sklearn", "networkx", "spacy", "openai")


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


def check_mcp_entry_point(*, bare: bool) -> None:
    mcp_cli = Path(sys.executable).parent / "mnemotree-mcp"
    assert mcp_cli.exists(), "mnemotree-mcp console script was not installed"
    if bare:
        # Without fastmcp and MemoryCore the server must refuse with an
        # install hint, not a bare traceback from somewhere deeper.
        proc = subprocess.run(
            [str(mcp_cli)], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60
        )
        assert proc.returncode != 0, "mnemotree-mcp started without fastmcp"
        assert "mnemotree[mcp_server]" in proc.stderr, proc.stderr
    else:
        # Starting the server would block on stdio; constructing it registers
        # every tool, which is where a broken import or signature shows up.
        from mnemotree.mcp import server

        assert server._get_mcp() is not None


async def evidence_roundtrip(workdir: Path) -> None:
    from mnemotree import Memory, Message

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


async def legacy_roundtrip(workdir: Path) -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from mnemotree import MemoryCore
    from mnemotree.core.memory import ModeDefaultsConfig, NerConfig
    from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

    dim = 32
    store = SQLiteVecMemoryStore(db_path=workdir / "memories.sqlite", embedding_dim=dim)
    await store.initialize()
    try:
        # Keywords and NER are off because both load spaCy's en_core_web_sm,
        # which is a separate download rather than a dependency. Lite mode
        # enables keywords by default, so a bare MemoryCore() on a fresh
        # install fails until the model is fetched; that is a product
        # question, not something this smoke test should paper over.
        core = MemoryCore(
            store=store,
            embeddings=DeterministicFakeEmbedding(size=dim),
            mode_defaults=ModeDefaultsConfig(enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )
        stored = await core.remember("The package smoke test stored this memory.")
        recalled = await core.recall("The package smoke test stored this memory.", limit=1)
        assert [m.memory_id for m in recalled] == [stored.memory_id], recalled
    finally:
        await store.close()


def main() -> None:
    bare = "--bare" in sys.argv
    version = importlib.metadata.version("mnemotree")

    check_location()
    if bare:
        for name in (*OPTIONAL_MODULES, *LEGACY_MODULES):
            assert importlib.util.find_spec(name) is None, f"Unexpected optional dependency: {name}"
    check_cli(version)
    check_evidence_cli()
    check_mcp_entry_point(bare=bare)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(evidence_roundtrip(Path(tmp)))
        if not bare:
            asyncio.run(legacy_roundtrip(Path(tmp)))
    print(f"mnemotree {version} ({'bare' if bare else 'mcp'}): ok")


if __name__ == "__main__":
    main()
