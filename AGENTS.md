# Mnemotree

## Project Summary

Mnemotree is a Python 3.11+ library giving LLM agents persistent, source-first memory. `EvidenceEngine` (in `src/mnemotree/engine.py`) stores original messages in SQLite with transactional ingestion, persistent FTS5 and sqlite-vec indexes, hybrid rank fusion, optional reranking, token-budget packing, and retrieval traces. The earlier `MemoryCore` and its storage backends were removed after 0.4.0; `mnemotree.Memory` binds one namespace over the engine, and new layers are built on top of it.

Key locations:
- Engine: `src/mnemotree/engine.py` (`EvidenceEngine`; `mnemotree.evidence.Memory` is its 1.0 name; `search` is a pipeline of `_plan_query`, `_retrieve`, `_rerank`, `pack`, `_trace`)
- Records, provider protocols and date annotations: `src/mnemotree/core/` (`models.py` including the atom records, `protocols.py` with `Embedder`, `Reranker`, `TokenCounter`, `Extractor`; `temporal.py`). Everything else depends on core, not on each other's internals
- Ingestion: `src/mnemotree/ingest/` (`chunking.py`; `atoms.py` batches passages and verifies each atom's cited quotes)
- Pure ranking steps (query bounding, RRF fusion, role policies, chronological order): `src/mnemotree/retrieval/ranking.py`
- Storage and indexes: `src/mnemotree/storage/` (`store.py`, `atom_store.py`, `lexical.py`; tables, version and embedding spec in `schema.py`; vector encoding in `vectors.py`)
- Golden retrieval snapshots: `tests/test_golden_retrieval.py`, `tests/golden/retrieval/` (regenerate with `MNEMOTREE_UPDATE_GOLDEN=1`)
- Packing and token counting: `src/mnemotree/retrieval/packing.py`
- Providers: `src/mnemotree/providers/` (`embeddings.py`, `extraction.py` with `OpenAIExtractor`, `retry.py`)
- Add/Search HTTP service: `src/mnemotree/service/` (`app.py`, `schemas.py`, `request_log.py`, `presets.py`)
- Evaluation and ablations: `src/mnemotree/evaluation.py`
- 1.0 import paths: `src/mnemotree/evidence/` only re-exports the moved modules, and nothing else in the package may import it; 2.0 deletes the directory (`tests/test_compat_paths.py`, which pins every 1.0 name)
- Public API: `src/mnemotree/memory.py` (`Memory`, `MemoryItem`, `Recall`)
- CLI/MCP JSON config, opened as an engine with `Config.open(path)`: `src/mnemotree/config.py`
- Exceptions (`MnemotreeError`, `InvalidInput`, `ConfigurationError`, `IngestConflict`, `ProviderError`): `src/mnemotree/errors.py`
- MCP stdio server: `src/mnemotree/mcp/` (standard library only), run as `mnemotree mcp`; the zero-config `mnemotree-mcp` package in `packages/mnemotree-mcp/` (a uv workspace member, released at the same version) adds fastembed and defaults to a local model (`FastEmbedEmbedder`, `DEFAULT_LOCAL_EMBEDDING` in `src/mnemotree/providers/embeddings.py`)
- CLI: `mnemotree-evidence` (`src/mnemotree/cli/evidence.py`), `mnemotree` with `mnemotree mcp` (`src/mnemotree/cli/__init__.py`), `mnemotree-mcp` (`packages/mnemotree-mcp/`)
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
