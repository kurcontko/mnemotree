# Mnemotree

## Project Summary

Mnemotree is a Python 3.10+ library for giving LLM agents biologically-inspired, persistent memory. The core API (`MemoryCore`) supports storing memories, recalling them via semantic similarity + filters, and optional analysis/enrichment (importance scoring, emotions, entities/keywords). It also ships a ready-to-run MCP server (`mnemotree-mcp`) so tools like Codex CLI / Claude Desktop can use the memory store over stdio or HTTP.

Key locations:
- Core API: `src/mnemotree/core/` (`MemoryCore`, models, retrieval/scoring)
- Storage backends: `src/mnemotree/store/` (ChromaDB, SQLite+sqlite-vec, Neo4j, Milvus)
- MCP server: `src/mnemotree/mcp/server.py`
- CLI entry points: `src/mnemotree/cli.py`
- Examples: `examples/` (Streamlit chat app in `examples/memory_chat/`)
- API stability policy: `docs/API.md` (public vs experimental internals)

## Common Commands

### Install (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional extras, e.g.:
# pip install -e ".[dev]"
# pip install -e ".[lite,chroma]"
```

### Install (uv)
```bash
uv venv
uv pip install -e .
# Optional extras, e.g.:
# uv pip install -e ".[dev]"
# uv pip install -e ".[lite,chroma]"
```

### Common Make targets
```bash
make lint
make typecheck
make test
make build
```
