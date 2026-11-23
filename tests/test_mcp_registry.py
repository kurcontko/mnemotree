"""packages/mnemotree-mcp/server.json is what the MCP Registry publishes.

The registry rejects a manifest whose version is not on PyPI, or whose name is
missing from the PyPI description, so both are pinned to the package here.
"""

import json
import re
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "packages" / "mnemotree-mcp"
MANIFEST = json.loads((PACKAGE / "server.json").read_text())
PROJECT = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]


def test_manifest_versions_match_the_package():
    (package,) = MANIFEST["packages"]
    assert package["identifier"] == PROJECT["name"]
    assert MANIFEST["version"] == package["version"] == PROJECT["version"]


def test_package_readme_carries_the_registry_name():
    # PyPI ownership check: the README must hold "mcp-name: <name>" followed by whitespace.
    readme = (PACKAGE / PROJECT["readme"]).read_text()
    assert re.search(rf"mcp-name: {re.escape(MANIFEST['name'])}(\s|$)", readme)


def test_manifest_fits_the_registry_limits():
    assert len(MANIFEST["description"]) <= 100
    assert re.fullmatch(r"[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+", MANIFEST["name"])


def test_manifest_lists_only_settings_the_server_reads():
    server = (PACKAGE.parents[1] / "src" / "mnemotree" / "mcp" / "server.py").read_text()
    (package,) = MANIFEST["packages"]
    for variable in package["environmentVariables"]:
        prefix, _, name = variable["name"].partition("_")
        assert prefix == "MNEMOTREE"
        assert f'setting("{name}")' in server or f'"{variable["name"]}"' in server
