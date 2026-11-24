#!/bin/sh
# Start the same executable for the MCP server and both Claude Code hooks.
set -f
hook=false
[ "${1:-}" = hook ] && hook=true
if [ -n "${MNEMOTREE_MCP_COMMAND:-}" ]; then
  if [ -x "$MNEMOTREE_MCP_COMMAND" ]; then
    set -- "$MNEMOTREE_MCP_COMMAND" "$@"
  else
    # The development override also accepts a command with arguments.
    set -- $MNEMOTREE_MCP_COMMAND "$@"
  fi
else
  set -- uvx --from "mnemotree-mcp>=1.2.0" mnemotree-mcp "$@"
fi

if [ "$hook" = false ]; then
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "mnemotree: cannot start $1. Install uv or set MNEMOTREE_MCP_COMMAND." >&2
    exit 127
  fi
  exec "$@"
fi

# Exit 2 vetoes a Claude prompt, even when it came from uv before Python started.
# Only a successful handler's stdout is hook output. Keep startup diagnostics too.
umask 077
state=${CLAUDE_PLUGIN_DATA:-"$HOME/.mnemotree/claude-code"}
run_hook() {
  if output=$("$@"); then
    [ -z "$output" ] || printf '%s\n' "$output"
  else
    printf 'mnemotree: hook launcher exited %s\n' "$?" >&2
  fi
}
if { [ -d "$state" ] || mkdir -p "$state"; } && ( : >> "$state/hook.log" ); then
  run_hook "$@" 2>> "$state/hook.log"
else
  run_hook "$@"
fi
exit 0
