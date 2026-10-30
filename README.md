# 🌳 Mnemotree

Local, source-first memory for LLM agents: SQLite, FTS5 and sqlite-vec.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![CI](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=kurcontko_mnemotree&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=kurcontko_mnemotree)
[![CodeQL](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml)

<p align="center">
  <img src="assets/mnemotree-logo.png" alt="Mnemotree Logo" width="300">
</p>

Mnemotree keeps every message exactly as written and recalls attributed source spans
instead of rewritten summaries.

- **Durable:** sources, chunks, both indexes and an idempotency receipt commit in one
  transaction.
- **Hybrid:** FTS5 BM25 and sqlite-vec cosine search, fused with RRF, with optional
  reranking.
- **Bounded:** evidence is packed into an exact token budget, with a trace of every
  candidate, rank and packing loss.
- **Isolated:** every read and write is scoped to one namespace.

The default install needs only Pydantic, sqlite-vec and tiktoken. Retrieval quality
has not been benchmarked yet.

## Install

```bash
pip install -e .
```

Python's `sqlite3` must support loadable extensions and FTS5. Tiktoken downloads its
vocabulary once; see [the guide](docs/GUIDE.md#configuration) to run offline.

| Extra | Adds |
| --- | --- |
| `embeddings` | OpenAI-compatible HTTP embedder and extractor |
| `lite` | Local SentenceTransformer embedder |
| `serve` | AML Add/Search HTTP service |
| `all` | All of the above |

## Quickstart

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

`Memory` binds one namespace and offers `remember`, `remember_many`, `recall`, `get`
and `forget`. Pass an `embedder` for hybrid retrieval.

## MCP server

`mnemotree-mcp` gives Claude Code, Claude Desktop, Codex and other MCP clients the
same five tools over stdio. It needs no extra.

```bash
claude mcp add mnemotree -- mnemotree-mcp --db ~/.mnemotree/memory.sqlite \
  --namespace "$USER" --lexical   # or --config with an embedding
```

Sessions pointed at the same `--db` share one memory.

## Command line

```bash
# Retrieval evaluation on the bundled plumbing example (needs a fresh database).
mnemotree-evidence --db demo.sqlite --config examples/evidence/lexical.json \
  evaluate examples/evidence/corpus.jsonl examples/evidence/queries.jsonl \
  --output demo-results.json

# AML Add/Search service with a bearer key (the serve extra).
export MNEMOTREE_API_KEY='replace-with-your-service-key'
mnemotree-evidence --db service.sqlite serve
```

`mnemotree-evidence` also runs `add`, `search`, and the `ablate`, `ablate-context` and
`ablate-retrieval` comparisons.

## Documentation

[docs/GUIDE.md](docs/GUIDE.md) covers the API contract, MCP configuration, embeddings,
storage and retrieval behavior, atomic projections, the AML service, evaluation
and limits.

Releases up to 0.4.0 shipped `MemoryCore`, pluggable vector/graph backends and an
older MCP server. They have been removed, and `Memory` does not read their databases.

## Development

```bash
uv sync --extra dev --extra serve
make lint typecheck test
make precommit-install
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
