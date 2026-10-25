"""Scope-budget ratchet: zone line counts and client-facing surface vs budget.toml.

Growing a zone past its ceiling fails here until the ceiling is raised (a
reviewable decision in budget.toml). Shrinking a zone more than 10% below its
ceiling also fails, so deletions ratchet the ceiling down instead of banking
invisible headroom.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "mnemotree"
BUDGET = tomllib.loads((REPO_ROOT / "budget.toml").read_text())

# Fraction of each ceiling that may remain available before a ratchet-down.
SLACK = 0.10

ZONE_DIRS = {
    "core": {"core"},
    "store": {"store"},
    "integrations": {"adapters", "integrations", "tools", "mcp"},
    "experimental": {"experimental"},
}


def _significant_lines(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(errors="replace").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _zone_of(path: Path) -> str:
    top = path.relative_to(PACKAGE).parts[0]
    return next((zone for zone, dirs in ZONE_DIRS.items() if top in dirs), "enrichment")


def _zone_totals() -> dict[str, int]:
    totals = dict.fromkeys(BUDGET["zones"], 0)
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" not in path.parts:
            totals[_zone_of(path)] += _significant_lines(path)
    return totals


def test_every_zone_has_a_ceiling() -> None:
    assert set(BUDGET["zones"]) == {*ZONE_DIRS, "enrichment"}


@pytest.mark.parametrize("zone", sorted(BUDGET["zones"]))
def test_zone_within_budget(zone: str) -> None:
    actual = _zone_totals()[zone]
    ceiling = BUDGET["zones"][zone]
    assert actual <= ceiling, (
        f"zone {zone!r} is {actual} significant lines, over its {ceiling}-line budget. "
        f"Cut {actual - ceiling} lines, or raise the ceiling in budget.toml in this PR "
        f"and defend the scope growth in review."
    )


@pytest.mark.parametrize("zone", sorted(BUDGET["zones"]))
def test_zone_ceiling_not_stale(zone: str) -> None:
    actual = _zone_totals()[zone]
    ceiling = BUDGET["zones"][zone]
    floor = int(ceiling * (1 - SLACK))
    assert actual >= floor, (
        f"zone {zone!r} is {actual} significant lines, more than {SLACK:.0%} under its "
        f"{ceiling}-line ceiling. Nice deletion — now ratchet the ceiling down in "
        f"budget.toml so the headroom doesn't quietly grow back."
    )


def test_mcp_tool_count_within_budget() -> None:
    from mnemotree.mcp import server

    registered: list[str] = []

    class _Recorder:
        def tool(self, fn):  # mirrors FastMCP.tool's decorator use
            registered.append(fn.__name__)
            return fn

    server._register_tools(_Recorder())
    max_tools = BUDGET["surface"]["max_mcp_tools"]
    assert len(registered) <= max_tools, (
        f"the MCP server registers {len(registered)} tools ({', '.join(registered)}); "
        f"budget.toml allows {max_tools}. Every client pays prompt tokens for every tool "
        f"on every session — fold the new capability into an existing tool's parameters, "
        f"or bump max_mcp_tools in this PR."
    )


def test_public_exports_within_budget() -> None:
    import mnemotree

    max_exports = BUDGET["surface"]["max_public_exports"]
    assert len(mnemotree.__all__) <= max_exports, (
        f"mnemotree.__all__ has {len(mnemotree.__all__)} names; budget.toml allows "
        f"{max_exports}. Each public name is a stability promise — export it from a "
        f"submodule instead, or bump max_public_exports in this PR."
    )
