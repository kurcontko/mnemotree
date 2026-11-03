# mnemotree-mcp

Long-term memory for Claude Code, Claude Desktop, Codex and other MCP clients,
with semantic search on a local embedding model. No API key or configuration is
needed:

```bash
claude mcp add mnemotree -- uvx mnemotree-mcp
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
