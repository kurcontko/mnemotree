"""The 1.0 module paths under mnemotree.evidence keep working after the move."""

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mnemotree
import mnemotree.evidence

MOVED = {
    "models": "mnemotree.core.models",
    "protocols": "mnemotree.core.protocols",
    "temporal": "mnemotree.core.temporal",
    "chunking": "mnemotree.ingest.chunking",
    "atoms": "mnemotree.ingest.atoms",
    "packing": "mnemotree.retrieval.packing",
    "store": "mnemotree.storage.store",
    "lexical": "mnemotree.storage.lexical",
    "atom_store": "mnemotree.storage.atom_store",
    "providers": "mnemotree.providers.embeddings",
    "retry": "mnemotree.providers.retry",
    "server": "mnemotree.service.app",
    "engine": "mnemotree.engine",
    "evaluation": "mnemotree.evaluation",
    "cli": "mnemotree.cli.evidence",
}
SRC = str(Path(mnemotree.__file__).parents[1])
# Every public name each 1.0 module defined or imported from the package, read from
# the 1.0.3 sources; none of them may stop importing before 2.0.
# Wrappers that keep a 1.0 behaviour the moved function changed: secret exits.
KEEPS_1_0_BEHAVIOUR = {("cli", "secret")}
NAMES_1_0 = json.loads(
    (Path(__file__).parent / "golden" / "compat" / "evidence-1.0-names.json").read_text()
)


def test_the_aliases_cover_every_1_0_module():
    assert set(NAMES_1_0) == set(MOVED)


@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_every_1_0_name_still_imports_from_its_old_path(old, new):
    module = importlib.import_module(f"mnemotree.evidence.{old}")
    assert getattr(mnemotree.evidence, old) is module
    missing = [name for name in NAMES_1_0[old] if not hasattr(module, name)]
    assert not missing
    # The same objects, not copies, so isinstance and identity checks keep holding.
    moved = importlib.import_module(new)
    for name in NAMES_1_0[old]:
        if hasattr(moved, name) and (old, name) not in KEEPS_1_0_BEHAVIOUR:
            assert getattr(module, name) is getattr(moved, name), name


def test_the_engine_keeps_its_1_0_name():
    from mnemotree.evidence.engine import Memory as OldEngineMemory

    assert mnemotree.EvidenceEngine is mnemotree.evidence.EvidenceEngine
    assert mnemotree.evidence.Memory is OldEngineMemory is mnemotree.EvidenceEngine
    # The namespace-bound API is a different class and keeps its own name.
    assert mnemotree.Memory is not mnemotree.EvidenceEngine


def test_only_the_aliases_import_the_1_0_paths():
    package = Path(mnemotree.__file__).parent
    offenders = []
    for path in package.rglob("*.py"):
        if path.parent.name == "evidence":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                target, names = "", [alias.name for alias in node.names]
            else:
                continue
            if "evidence" in target.split(".") or any(
                name.startswith("mnemotree.evidence") or name == "evidence" for name in names
            ):
                offenders.append(f"{path.relative_to(package)}:{node.lineno}")
    assert not offenders


def run(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )


def test_importing_the_package_loads_no_optional_extra():
    probe = run(
        "import sys, mnemotree, mnemotree.evidence, mnemotree.mcp.server\n"
        "loaded = [m for m in ('fastapi', 'uvicorn', 'httpx', 'mnemotree.service.app')"
        " if m in sys.modules]\n"
        "assert not loaded, loaded"
    )
    assert probe.returncode == 0, probe.stderr


def test_old_cli_module_still_runs_as_a_script():
    env = {**os.environ, "PYTHONPATH": SRC}
    result = subprocess.run(
        [sys.executable, "-m", "mnemotree.evidence.cli", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "serve" in result.stdout
