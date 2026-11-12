"""Each package of mnemotree imports only the packages below it."""

import ast
from pathlib import Path

import pytest

import mnemotree

ROOT = Path(mnemotree.__file__).parent

# What each top-level package or module may import from the rest of mnemotree.
# The CLIs sit on top, and ``evidence`` (the 1.0 paths) re-exports from everywhere.
LOWER = {"core", "errors"}
ALLOWED = {
    "errors": set(),
    "core": {"errors"},
    "ingest": LOWER,
    "retrieval": LOWER,
    "storage": LOWER,
    "providers": LOWER,
    "engine": LOWER | {"ingest", "retrieval", "storage"},
    "memory": LOWER | {"engine"},
    "evaluation": LOWER | {"engine"},
    "service": LOWER | {"engine"},
    "config": LOWER | {"engine", "providers", "retrieval"},
    "mcp": LOWER | {"engine", "memory", "config", "providers"},
}


def imported_packages(path: Path) -> set[str]:
    """Top-level mnemotree packages that the module at ``path`` imports."""
    package = ("mnemotree", *path.relative_to(ROOT).parts[:-1])
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(package[: len(package) - node.level + 1]) if node.level else ""
            module = ".".join(part for part in (base, node.module) if part)
            # ``from . import engine`` names the package in the alias, not the module.
            names = [f"{module}.{alias.name}" for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == "mnemotree" and len(parts) > 1:
                found.add(parts[1])
    return found


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_package_imports_only_lower_layers(layer):
    module = ROOT / f"{layer}.py"
    paths = [module] if module.exists() else sorted((ROOT / layer).rglob("*.py"))
    assert paths, f"no source found for {layer}"
    known = {path.stem if path.is_file() else path.name for path in ROOT.iterdir()}
    for path in paths:
        outside = (imported_packages(path) & known) - {layer} - ALLOWED[layer]
        assert not outside, f"{path.relative_to(ROOT)} imports {sorted(outside)}"


def test_every_package_has_a_rule():
    packages = {
        path.stem if path.is_file() else path.name
        for path in ROOT.iterdir()
        if path.suffix == ".py" or (path / "__init__.py").exists()
    }
    assert packages - {"__init__", "cli", "evidence"} == set(ALLOWED)
