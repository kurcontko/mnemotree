# 🌳 Mnemotree

Local evidence memory for LLM agents: SQLite, FTS5, and sqlite-vec.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![CI](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=kurcontko_mnemotree&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=kurcontko_mnemotree)
[![CodeQL](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml)

<p align="center">
  <img src="assets/mnemotree-logo.png" alt="Mnemotree Logo" width="300">
</p>

`Memory` preserves original messages and retrieves attributed source spans. It provides transactional ingestion, persistent lexical/vector indexes,
hybrid rank fusion, optional reranking, token-budget packing, and retrieval traces.
This is a baseline for measured experiments, with no benchmark-quality claim yet.

## Quickstart

```bash
pip install -e .
```

```python
import asyncio
from mnemotree import Memory

async def main():
    with Memory("memory.sqlite", namespace="alice") as memory:  # lexical without an embedder
        await memory.remember("I grow orchids on my balcony.", role="user")
        recall = await memory.recall("orchids")
        print(recall.context)               # packed, attributed evidence
        print([item.id for item in recall.items], recall.trace.packed_tokens)

asyncio.run(main())
```

`Memory` binds one namespace and offers `remember`, `remember_many` (ordered turns of
one session), `recall`, `get` and `forget`; see [the API guide](docs/MEMORY_API.md).
The engine underneath, with per-call namespaces and every retrieval option, is
`mnemotree.evidence.Memory`.

The default install needs Pydantic, sqlite-vec, and tiktoken. Python must support
SQLite loadable extensions and FTS5. Tiktoken downloads its vocabulary once;
prewarm and preserve `TIKTOKEN_CACHE_DIR` when running offline.

```bash
# Small plumbing demonstration; requires a fresh database path.
mnemotree-evidence --db demo.sqlite --config examples/evidence/lexical.json \
  evaluate examples/evidence/corpus.jsonl examples/evidence/queries.jsonl \
  --output demo-results.json

# Textual AML Add/Search service, authenticated with a bearer key.
pip install -e '.[serve]'
export MNEMOTREE_API_KEY='replace-with-your-service-key'
mnemotree-evidence --db service.sqlite serve
```

See [the evidence core guide](docs/EVIDENCE_CORE.md) for embeddings, API contracts,
evaluation format, and current limits. Every query requires an exact namespace;
search spans all sessions within it.

Optional [atomic search projections](docs/ATOMIC_MEMORY.md) and a three-way
`ablate` command are available. Raw-only behavior remains the default.

Earlier releases (0.4.0 and before) shipped `MemoryCore`, pluggable vector/graph
backends and an MCP server. They were removed so the library can be rebuilt
around this engine; their databases are not read by `Memory`.

## MCP server

`mnemotree-mcp` exposes `remember`, `remember_many`, `recall`, `get` and `forget` over
stdio to Claude Code, Claude Desktop, Codex and other MCP clients. It needs no extra.

```bash
claude mcp add mnemotree -- mnemotree-mcp --db ~/.mnemotree/memory.sqlite \
  --namespace "$USER" --lexical   # or --config with an embedding
```

Sessions pointed at the same `--db` share one memory. See
[the API guide](docs/MEMORY_API.md#mcp-server) for configuration.

## 📦 Extras

```bash
uv pip install -e ".[embeddings]"  # OpenAI-compatible HTTP embedder
uv pip install -e ".[lite]"        # Local SentenceTransformer embedder
uv pip install -e ".[serve]"       # AML Add/Search HTTP adapter
uv pip install -e ".[all]"         # Everything above
```

## Development

```bash
uv sync --extra dev --extra serve
make lint typecheck test
make precommit-install
```

## 🤝 Contributing

Contributions welcome! Fork the repo, create a branch, add tests, and submit a PR.

## 📝 License

MIT - see [LICENSE](LICENSE)
