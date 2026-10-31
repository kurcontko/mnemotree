# Mnemotree

## Project Summary

Mnemotree is a Python 3.11+ library giving LLM agents persistent, source-first memory. `Memory` (in `src/mnemotree/evidence/`) stores original messages in SQLite with transactional ingestion, persistent FTS5 and sqlite-vec indexes, hybrid rank fusion, optional reranking, token-budget packing, and retrieval traces. The earlier `MemoryCore` and its storage backends were removed after 0.4.0; `mnemotree.Memory` binds one namespace over the engine, and new layers are built on top of it.

Key locations:
- Engine: `src/mnemotree/evidence/engine.py` (`Memory`), models in `models.py`, protocols in `protocols.py`
- Storage and indexes: `src/mnemotree/evidence/store.py`, `atom_store.py`, `lexical.py`
- Packing and context: `src/mnemotree/evidence/packing.py`, `temporal.py`
- Providers (embedders, rerankers): `src/mnemotree/evidence/providers.py`
- Add/Search HTTP service: `src/mnemotree/evidence/server.py`
- Public API: `src/mnemotree/memory.py` (`Memory`, `MemoryItem`, `Recall`)
- CLI/MCP JSON config and the factory that opens an engine from it: `src/mnemotree/config.py`
- Exceptions (`MnemotreeError`, `InvalidInput`, `ConfigurationError`, `IngestConflict`): `src/mnemotree/errors.py`
- MCP stdio server: `src/mnemotree/mcp/` (`mnemotree-mcp`, standard library only)
- CLI: `mnemotree-evidence` (`src/mnemotree/evidence/cli.py`), `mnemotree-mcp`, `mnemotree` (`src/mnemotree/cli.py`)
- Guide: `docs/GUIDE.md` (API, MCP, configuration, retrieval, atomic projections, AML service, evaluation, limits)

## Common Commands

### Install (uv)
```bash
uv sync --extra dev --extra serve
# Optional extras: embeddings (HTTP embedder), lite (local SentenceTransformer), serve (HTTP service)
```

### Install (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,serve]"
```

### Common Make targets
```bash
make lint
make typecheck
make test
make build
```
