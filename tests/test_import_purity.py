"""mnemotree must import without its optional backends.

`pip install mnemotree` ships the core, the SQLite store, and the lazy
entry points; chromadb, neo4j, milvus, local models, fastmcp and the other
extras arrive only when asked for. This test enforces that boundary whatever
the dev environment has installed: a meta-path hook makes every optional
package unimportable in a subprocess, then imports every module. A
module-level `import chromadb` sneaking into a shared module fails here
before it fails for a bare-install user.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "mnemotree"

# Top-level names of every optional dependency (pyproject extras), plus the
# transitive `mcp` that fastmcp brings and `autogen`, which tools/autogen.py
# imports without any extra declaring it.
OPTIONAL = [
    "autogen",
    "chromadb",
    "dateparser",
    "fastmcp",
    "flashrank",
    "gliner",
    "langchain_anthropic",
    "langchain_aws",
    "langchain_google_genai",
    "langchain_huggingface",
    "mcp",
    "neo4j",
    "pymilvus",
    "pyspark",
    "sentence_transformers",
    "sparknlp",
    "stanza",
    "streamlit",
    "torch",
    "transformers",
    "yake",
]

# Modules whose whole purpose is one optional backend. Everything else must
# import bare. Adding a module here is a decision to make in review.
BACKEND_MODULES = {
    "mnemotree.store.baseline.chromadb_store",
    "mnemotree.store.baseline.neo4j_store",
    "mnemotree.store.chroma_utils",
    "mnemotree.store.chromadb_store",
    "mnemotree.store.milvus_store",
    "mnemotree.store.neo4j_store",
    "mnemotree.tools.autogen",
}

BLOCKER = """
import importlib
import sys

class _BlockOptional:
    BLOCKED = set(%r)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)", name=name)
        return None

sys.meta_path.insert(0, _BlockOptional())
"""


def _all_modules() -> list[str]:
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCKER % (OPTIONAL,) + code],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_backend_allowlist_is_current() -> None:
    stale = BACKEND_MODULES - set(_all_modules())
    assert not stale, f"BACKEND_MODULES names modules that no longer exist: {sorted(stale)}"


def test_public_exports_resolve() -> None:
    import mnemotree

    for name in mnemotree.__all__:
        assert getattr(mnemotree, name) is not None


def test_every_shared_module_imports_without_optional_deps() -> None:
    modules = [m for m in _all_modules() if m not in BACKEND_MODULES]
    proc = _run(
        f"failed = []\n"
        f"for mod in {modules!r}:\n"
        f"    try:\n"
        f"        importlib.import_module(mod)\n"
        f"    except Exception as exc:\n"
        f"        failed.append(f'{{mod}}: {{type(exc).__name__}}: {{exc}}')\n"
        f"print('\\n'.join(failed) or 'ok')\n"
    )
    report = f"{proc.stdout}{proc.stderr}"
    assert proc.returncode == 0, report
    assert proc.stdout.strip() == "ok", (
        f"modules pulled in an optional dependency at import time:\n{report}"
    )


def test_lazy_store_export_names_the_extra() -> None:
    proc = _run(
        "import mnemotree.store as store\n"
        "try:\n"
        "    store.ChromaMemoryStore\n"
        "except ModuleNotFoundError as exc:\n"
        "    print(exc)\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "Install with" in proc.stdout, proc.stdout
    assert "mnemotree[chroma]" in proc.stdout, proc.stdout


def test_mcp_entry_point_hints_at_extra_when_fastmcp_missing() -> None:
    proc = _run("from mnemotree.mcp.server import main\nmain()\n")
    assert proc.returncode != 0
    assert "mnemotree[mcp_server]" in proc.stderr, proc.stderr
