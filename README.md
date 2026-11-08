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

The default install needs only Pydantic, sqlite-vec and tiktoken. See
[Benchmarks](#benchmarks) for end-to-end results on LoCoMo and LongMemEval.

## Install

```bash
pip install mnemotree
```

This README describes the 0.5 API. When using it before the 0.5 release, install
the development version instead:

```bash
pip install "mnemotree @ git+https://github.com/kurcontko/mnemotree.git@main"
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

## Benchmarks

These scores measure end-to-end question answering with the `v1.0.0` engine.
Mnemotree packs at most 8,000 tokens of evidence per question. gpt-4o-mini answers
from that evidence and also judges the answer. Embeddings come from DashScope
`text-embedding-v4` at 1,024 dimensions. Retrieval fuses BM25 and vector search
with RRF and adds the neighboring messages around each hit. No LLM runs inside
Mnemotree.

| Benchmark | Questions | Mnemotree | Published, gpt-4o-mini answering |
| --- | ---: | ---: | --- |
| LoCoMo, categories 1 to 4 | 1,540 | 72.1 | Full context (26K tokens) 72.9, Mem0 with graph 68.4, Mem0 66.9, Zep 66.0, LangMem 58.1 ([Mem0 paper](https://arxiv.org/abs/2504.19413), Table 2) |
| LongMemEval-S | 500 | 60.8 | Zep 63.8, full context (115K tokens) 55.4 ([Zep paper](https://arxiv.org/abs/2501.13956), Table 2) |

On LoCoMo-Refined, a cleaned release of LoCoMo with 1,382 questions, Mnemotree
scores 78.7 with the [official scorer](https://github.com/mem-eval-suite/LoCoMo_refined),
which judges with Qwen3-14B. That run used Nemotron-3-Embed-8B embeddings.

Each score is a single run of ours. The published rows are the authors' own runs
with their own prompts and judges, and the Zep paper judges LongMemEval with gpt-4o.
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) has the setup, the method, how to size the
evidence budget and the limitations. The evaluation harness is not in this
repository yet.

## MCP server

`mnemotree-mcp` gives Claude Code, Claude Desktop, Codex and other MCP clients the
same five tools over stdio, with semantic search on a local embedding model. It
needs no API key or configuration:

```bash
claude mcp add mnemotree -- uvx mnemotree-mcp
```

The first start downloads the English-only
[granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
model (about 200 MB) to `~/.mnemotree/models`; the server answers while it loads.

| Setting | Flag | Default |
| --- | --- | --- |
| `MNEMOTREE_DB` | `--db` | `~/.mnemotree/memory.sqlite` |
| `MNEMOTREE_NAMESPACE` | `--namespace` | `default` |
| `MNEMOTREE_CONFIG` | `--config` | local model |
| `MNEMOTREE_LEXICAL=1` | `--lexical` | off |

```bash
uvx mnemotree-mcp --lexical                    # keyword-only, no model download
uvx mnemotree-mcp --config ~/.mnemotree/config.json   # e.g. an HTTP embedder
```

Sessions pointed at the same database share one memory. With `pip install mnemotree`,
`mnemotree mcp` runs the same server, but with no default model: pass `--config` or
`--lexical`.

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
and limits. [docs/BENCHMARKS.md](docs/BENCHMARKS.md) records how the benchmark
numbers were produced and what they leave out.

Releases up to 0.4.0 shipped `MemoryCore`, pluggable vector/graph backends and an
older MCP server. They have been removed, and `Memory` does not read their databases.
See the [0.5 release notes](docs/RELEASE_NOTES.md) for API and compatibility changes.

## Development

```bash
git clone https://github.com/kurcontko/mnemotree.git
cd mnemotree
uv sync --extra dev --extra serve
make lint typecheck test
make build
make precommit-install
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
