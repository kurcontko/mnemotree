# 🌳 Mnemotree

Memory module for LLMs and Agents with MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=kurcontko_mnemotree&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=kurcontko_mnemotree)
[![CodeQL](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml)

<p align="center">
  <img src="assets/mnemotree-logo.png" alt="Mnemotree Logo" width="300">
</p>

Mnemotree gives LLM agents biologically-inspired memory. Store, retrieve, and analyze structured knowledge with semantic search, importance scoring, and relationship tracking. Integrates with LangChain, Autogen, and any MCP-compliant tool.

## ⚡ MCP Quickstart

Run mnemotree as an MCP server with zero setup:

```bash
uvx --from "git+https://github.com/kurcontko/mnemotree.git" --with "mnemotree[mcp_server]" mnemotree-mcp
```

### Claude Desktop / Claude Code

Add to your config (`claude_desktop_config.json`, `.mcp.json`, or `~/.claude.json`):

```json
{
  "mcpServers": {
    "mnemotree": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/kurcontko/mnemotree.git",
        "--with", "mnemotree[mcp_server]",
        "mnemotree-mcp"
      ],
      "env": {
        "MNEMOTREE_MCP_PERSIST_DIR": "/Users/yourname/.mnemotree/chromadb"
      }
    }
  }
}
```

### Codex CLI

Add to your `~/.codex/config.toml`:

```toml
[mcp_servers.mnemotree]
command = "uvx"
args = [
  "--from", "git+https://github.com/kurcontko/mnemotree.git",
  "--with", "mnemotree[mcp_server]",
  "mnemotree-mcp",
]
startup_timeout_sec = 120
env = { MNEMOTREE_MCP_PERSIST_DIR = "/Users/yourname/.mnemotree/chromadb" }
```

### Local Development

Replace `git+https://...` with `/path/to/mnemotree` to use your local clone.

### Persistence

`MNEMOTREE_MCP_PERSIST_DIR` controls where memories are stored. Use an **absolute path** for consistent storage across clients. Omit to default to `.mnemotree/chromadb`.

### HTTP Transport (Multi-Client)

```bash
uvx --from "git+https://github.com/kurcontko/mnemotree.git" --with "mnemotree[mcp_server]" mnemotree-mcp run --transport http --port 8000
```

Connect MCP clients to `http://localhost:8000/mcp`.


## 🌟 Features

- **Memory Types**: Episodic, semantic, autobiographical, prospective, procedural, priming, conditioning, working, entities
- **Storage Backends**: ChromaDB, SQLite+sqlite-vec, Neo4j
- **Analysis**: NER, keyword extraction, importance scoring, emotional context
- **Retrieval**: Semantic similarity, filtering, relationship queries
- **Lite Mode**: CPU-only embeddings, no LLM required

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/kurcontko/mnemotree.git && cd mnemotree
uv venv .venv && uv pip install -e ".[lite,chroma]"
```

For NER: `uv run python -m spacy download en_core_web_sm`

For OpenAI features: `cp .env.sample .env` and add your API key.

### Basic Usage

```python
from mnemotree import MemoryCore
from mnemotree.store import ChromaMemoryStore

store = ChromaMemoryStore(persist_directory=".mnemotree/chromadb")
memory_core = MemoryCore(store=store)

# Store
memory = await memory_core.remember(
    content="User prefers Python for its readability.",
    tags=["preferences", "programming"]
)

# Recall
memories = await memory_core.recall("programming languages", limit=5)

# Reflect
insights = await memory_core.reflect(min_importance=0.7)
```

### Lite Mode (CPU, no LLM)

```python
memory_core = MemoryCore(store=store, mode="lite")
```

Uses local embeddings. Set `MNEMOTREE_LITE_EMBEDDING_MODEL` to override.

Alternative NER backends: `mnemotree[ner_hf]`, `mnemotree[ner_gliner]`, `mnemotree[ner_stanza]`

## ⚙️ MCP Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOTREE_MCP_PERSIST_DIR` | `.mnemotree/chromadb` | Storage directory |
| `MNEMOTREE_MCP_COLLECTION` | `memories` | Collection name |
| `MNEMOTREE_MCP_CHROMA_HOST/PORT/SSL` | — | Remote ChromaDB |
| `MNEMOTREE_MCP_ENABLE_NER` | `false` | Enable NER |
| `MNEMOTREE_MCP_ENABLE_KEYWORDS` | `false` | Enable keyword extraction |
| `MNEMOTREE_MCP_NER_BACKEND` | — | `spacy`, `transformers`, `gliner`, `stanza` |
| `MNEMOTREE_MCP_NER_MODEL` | — | Backend-specific model ID/path |

Avoid running multiple MCP processes against the same Chroma directory.

## 🔧 Storage

```python
# ChromaDB (local)
from mnemotree.store import ChromaMemoryStore
store = ChromaMemoryStore(persist_directory=".mnemotree/chromadb")

# ChromaDB (remote)
store = ChromaMemoryStore(host="localhost", port=8000)

# Neo4j
from mnemotree.store import Neo4jMemoryStore
store = Neo4jMemoryStore(uri="neo4j://localhost:7687", user="neo4j", password="password")
```

## 🐳 Docker

```bash
# MCP server
docker compose -f docker/mcp/docker-compose.yml up --build

# ChromaDB
docker compose -f docker/chromadb/docker-compose.yml up -d

# Neo4j
docker compose -f docker/neo4j/docker-compose.yml up -d
```

## 📦 Extras

```bash
uv pip install -e ".[chroma]"      # ChromaDB
uv pip install -e ".[neo4j]"       # Neo4j
uv pip install -e ".[sqlite_vec]"  # SQLite + sqlite-vec
uv pip install -e ".[lite]"        # Local embeddings
uv pip install -e ".[ner_hf]"      # Transformers NER
uv pip install -e ".[all]"         # Everything
```

## Development

```bash
make lint typecheck test
make precommit-install
```

## 💡 Examples

- [`examples/langchain_agent.py`](examples/langchain_agent.py) — LangChain agent with memory
- [`examples/memory_chat/app.py`](examples/memory_chat/app.py) — Streamlit chat app with persistent memory

## 🤝 Contributing

Contributions welcome! Fork the repo, create a branch, add tests, and submit a PR.

## 📝 License

MIT - see [LICENSE](LICENSE)
