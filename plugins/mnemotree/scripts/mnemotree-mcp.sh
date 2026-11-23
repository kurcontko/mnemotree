#!/bin/sh
# Starts mnemotree-mcp for the plugin's MCP server and for its hooks, so both run
# the same version against the same database.
#
# MNEMOTREE_MCP_COMMAND replaces the command, for example with a development
# checkout: MNEMOTREE_MCP_COMMAND="/path/to/.venv/bin/mnemotree-mcp".
if [ -n "${MNEMOTREE_MCP_COMMAND:-}" ]; then
  # Unquoted on purpose: the variable may hold a command with arguments.
  exec $MNEMOTREE_MCP_COMMAND "$@"
fi
if ! command -v uvx >/dev/null 2>&1; then
  # A hook must not fail the session over a missing tool. The MCP server does
  # fail, which is where Claude Code shows the reason.
  [ "${1:-}" = "hook" ] && exit 0
  echo "mnemotree: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 127
fi
exec uvx --from "mnemotree-mcp>=1.2.0" mnemotree-mcp "$@"
