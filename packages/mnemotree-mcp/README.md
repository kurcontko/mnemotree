# mnemotree-mcp

Long-term memory for Claude Code, Claude Desktop, Codex and other MCP clients,
with semantic search on a local embedding model. No API key or configuration is
needed:

```bash
claude mcp add mnemotree -- uvx mnemotree-mcp   # Claude Code
codex mcp add mnemotree -- uvx mnemotree-mcp    # Codex
pi mcp add mnemotree -- uvx mnemotree-mcp       # Pi
```

Claude Desktop, Cursor and VS Code take the same command as a JSON entry. The guide
lists [the file each client reads](https://github.com/kurcontko/mnemotree/blob/main/docs/GUIDE.md#client-setup), and the entry for OpenCode.

```json
{
  "mcpServers": {
    "mnemotree": {
      "command": "uvx",
      "args": ["mnemotree-mcp"]
    }
  }
}
```

The first start downloads
[granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
(about 200 MB, English only) to `~/.mnemotree/models`. The server answers at once
while the model loads; the first call that needs it waits.

Memories go to `~/.mnemotree/memory.sqlite`, namespace `default`. Change them with
`--db` and `--namespace`, or `MNEMOTREE_DB` and `MNEMOTREE_NAMESPACE`. Pass
`--lexical` (or `MNEMOTREE_LEXICAL=1`) for keyword-only search without the model,
or `--config` (`MNEMOTREE_CONFIG`) for an HTTP embedding provider.

This package is a thin wrapper: the server, its tools and its options are part of
[mnemotree](https://github.com/kurcontko/mnemotree). See its
[guide](https://github.com/kurcontko/mnemotree/blob/main/docs/GUIDE.md#mcp-server).

mcp-name: io.github.kurcontko/mnemotree
