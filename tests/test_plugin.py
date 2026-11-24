"""The Claude Code plugin in plugins/mnemotree and the marketplace that lists it."""

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "mnemotree"
MANIFEST = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
LAUNCHER = PLUGIN / "scripts" / "mnemotree-mcp.sh"


def version(text):
    return tuple(int(part) for part in text.split("."))


def test_marketplace_lists_the_plugin_at_its_directory():
    marketplace = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    (entry,) = marketplace["plugins"]
    assert entry["name"] == MANIFEST["name"] == "mnemotree"
    assert (ROOT / entry["source"]).resolve() == PLUGIN


def test_the_launcher_requires_the_release_the_plugin_was_built_for():
    # The hooks call `mnemotree-mcp hook`, which older releases do not have.
    (pin,) = re.findall(r'--from "mnemotree-mcp>=([\d.]+)"', LAUNCHER.read_text())
    assert pin == MANIFEST["version"]
    package = tomllib.loads((ROOT / "packages/mnemotree-mcp/pyproject.toml").read_text())
    # A release that forgets to raise the plugin version fails here.
    assert version(MANIFEST["version"]) >= version(package["project"]["version"])


def test_hooks_and_the_mcp_server_start_through_the_launcher():
    assert os.access(LAUNCHER, os.X_OK)
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {"SessionStart", "UserPromptSubmit", "Stop"}
    # Only a compaction or /clear changes what the hooks may recall.
    assert [group.get("matcher") for group in hooks["SessionStart"]] == ["compact|clear"]
    for groups in hooks.values():
        ((handler,),) = (group["hooks"] for group in groups)
        assert (
            handler["command"] == '"${CLAUDE_PLUGIN_ROOT}/scripts/mnemotree-mcp.sh" hook --project'
        )
    (server,) = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"].values()
    assert server == {
        "command": "${CLAUDE_PLUGIN_ROOT}/scripts/mnemotree-mcp.sh",
        "args": ["--project"],
    }


def test_launcher_runs_the_override_command_with_its_arguments(tmp_path):
    env = {
        "PATH": os.environ["PATH"],
        "CLAUDE_PLUGIN_DATA": str(tmp_path),
        "MNEMOTREE_MCP_COMMAND": "echo started with",
    }
    out = subprocess.run(
        [LAUNCHER, "hook", "--project"], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "started with hook --project"


def test_launcher_without_uv_fails_the_server_but_not_a_hook(tmp_path):
    env = {"PATH": str(tmp_path), "CLAUDE_PLUGIN_DATA": str(tmp_path)}
    hook = subprocess.run(["/bin/sh", LAUNCHER, "hook"], env=env, capture_output=True, text=True)
    assert (hook.returncode, hook.stdout, hook.stderr) == (0, "", "")
    server = subprocess.run(["/bin/sh", LAUNCHER], env=env, capture_output=True, text=True)
    assert server.returncode == 127
    assert "uvx" in server.stderr


def test_launcher_keeps_exit_two_and_partial_output_out_of_hook_responses(tmp_path):
    broken = tmp_path / "failing uvx"
    broken.write_text("#!/bin/sh\necho partial\necho unavailable >&2\nexit 2\n")
    broken.chmod(0o755)
    env = {**os.environ, "CLAUDE_PLUGIN_DATA": str(tmp_path), "MNEMOTREE_MCP_COMMAND": str(broken)}
    hook = subprocess.run([LAUNCHER, "hook", "--project"], env=env, capture_output=True, text=True)
    assert (hook.returncode, hook.stdout, hook.stderr) == (0, "", "")
    log = (tmp_path / "hook.log").read_text()
    assert "unavailable" in log and "exited 2" in log
    server = subprocess.run([LAUNCHER, "--project"], env=env, capture_output=True, text=True)
    assert (server.returncode, server.stdout, server.stderr) == (2, "partial\n", "unavailable\n")
