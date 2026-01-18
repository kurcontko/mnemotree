# 🌳 Mnemotree

An Advanced Memory Management and Retrieval System for LLM Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=kurcontko_mnemotree&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=kurcontko_mnemotree)
[![CodeQL](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml/badge.svg)](https://github.com/kurcontko/mnemotree/actions/workflows/codeql.yml)

<p align="center">
  <img src="assets/mnemotree-logo.png" alt="Mnemotree Logo" width="300">
</p>

## Overview

Mnemotree is a framework that enhances Large Language Model (LLM) agents with biologically-inspired memory capabilities. It provides a unified system for storing, retrieving, and analyzing structured knowledge representations, designed to seamlessly integrate with popular LLM frameworks like LangChain and Autogen.

It includes a ready-to-use **Model Context Protocol (MCP)** server, allowing easy integration with Claude Desktop, Codex CLI, and other MCP-compliant tools with a single command.

## ⚡ MCP Quickstart

Run mnemotree as an MCP server with zero setup:

```bash
uvx --from "git+https://github.com/kurcontko/mnemotree.git" --with "mnemotree[mcp_server]" mnemotree-mcp
```

### Claude Desktop / Claude Code

Claude Desktop uses `claude_desktop_config.json` (Settings -> Developer -> Edit Config).
Claude Code uses `.mcp.json` (project scope) or `~/.claude.json` (user/local scope).

Add to your config:

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

When developing locally, point to your cloned repository instead of GitHub:

**Claude Desktop (`claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "mnemotree": {
      "command": "uvx",
      "args": [
        "--from", "/path/to/mnemotree",
        "--with", "mnemotree[mcp_server]",
        "mnemotree-mcp"
      ],
      "env": {
        "MNEMOTREE_MCP_PERSIST_DIR": "/path/to/mnemotree/.mnemotree/chromadb"
      }
    }
  }
}
```

**Claude Code (`.mcp.json` or `~/.claude.json`):**

```json
{
  "mcpServers": {
    "mnemotree": {
      "command": "uvx",
      "args": [
        "--from", "/path/to/mnemotree",
        "--with", "mnemotree[mcp_server]",
        "mnemotree-mcp"
      ],
      "env": {
        "MNEMOTREE_MCP_PERSIST_DIR": "/path/to/mnemotree/.mnemotree/chromadb"
      }
    }
  }
}
```

**Codex CLI (`~/.codex/config.toml`):**

```toml
[mcp_servers.mnemotree]
command = "uvx"
args = [
  "--from", "/path/to/mnemotree",
  "--with", "mnemotree[mcp_server]",
  "mnemotree-mcp",
]
startup_timeout_sec = 120
env = { MNEMOTREE_MCP_PERSIST_DIR = "/path/to/mnemotree/.mnemotree/chromadb" }
```

### Persistence Directory

The `MNEMOTREE_MCP_PERSIST_DIR` environment variable controls where memories are stored on disk.

| Value | Behavior |
|-------|----------|
| Absolute path (e.g., `/Users/yourname/.mnemotree/chromadb`) | Memories persist across sessions in this directory |
| Relative path (e.g., `.mnemotree/chromadb`) | Created relative to working directory (may vary per client) |
| Omitted | Defaults to `.mnemotree/chromadb` in the working directory |

**Recommendations:**
- Use an **absolute path** for consistent storage across all clients
- Use a **shared directory** (e.g., `~/.mnemotree/chromadb`) if you want Claude Desktop and Codex to share the same memory store
- Use **separate directories** per tool if you want isolated memory stores

### HTTP Transport (Multi-Client)

For multiple clients sharing one memory store, run the server as a separate process:

```bash
uvx --from "git+https://github.com/kurcontko/mnemotree.git" \
    --with "mnemotree[mcp_server]" \
    mnemotree-mcp run --transport http --port 8000
```

Then connect MCP clients to `http://localhost:8000/mcp`.


## 🌟 Key Features

### Memory Management
- **Multiple Memory Types**: Support for various memory categories including:
  - Episodic (personal experiences)
  - Semantic (facts and knowledge)
  - Autobiographical (personal life story)
  - Procedural (skills and procedures)
  - Working (short-term processing)

### Advanced Analysis
- **Contextual Processing**: Automatic analysis of memory content using LLMs
- **Emotional Context**: Detection and storage of emotional valence and arousal
- **Importance Scoring**: Dynamic scoring system considering:
  - Recency
  - Access frequency
  - Emotional weight
  - Contextual relevance
  - Memory type

### Storage & Retrieval
- **Flexible Storage Backend**:
  - ChromaDB integration for vector similarity search
  - SQLite + sqlite-vec for single-file local vector storage
  - Neo4j support for graph-based relationships
  - Extensible architecture for custom storage implementations
- **Advanced Querying**:
  - Semantic similarity search
  - Complex filtering
  - Relationship-based queries
  - Importance-based filtering
- **MCP Server (FastMCP)**:
  - Lightweight MCP wrapper for `MemoryCore`
  - Works with `uvx` and a single shared server for multiple clients

## 🚀 Getting Started

### Prerequisites

```bash
- Python 3.10+
- uv for dependency management (https://github.com/astral-sh/uv)
- Access to an LLM API (e.g., OpenAI)
- Vector store (ChromaDB or SQLite + sqlite-vec) or graph database (Neo4j) for storage
```

**Download and Install spaCy's English Model**

To use the default Named Entity Recognition (NER) functionality provided by spaCy, you need to install the English language model. Run the following command:

```bash
uv run python -m spacy download en_core_web_sm
```

**Set up OpenAI API or other LLM provider Credentials**

If you plan to use the default OpenAI-based implementation for language-related features, you need to provide your OpenAI API key. To do this:

Create a copy of the provided sample environment file:

```bash
cp .env.sample .env
```
Open the newly created .env file and replace the placeholder with your actual OpenAI API key:

```env
OPENAI_API_KEY=your-openai-api-key
```

### Installation

```bash
# Default mode is `lite` (local embeddings) which requires the `lite` extra.
# The README examples below also use the Chroma store.
uv pip install -e ".[lite,chroma]"
```

### Basic Usage

```python
from mnemotree import MemoryCore
from mnemotree.store import ChromaMemoryStore
from mnemotree.core.models import MemoryType

# Initialize with ChromaDB storage
store = ChromaMemoryStore(persist_directory=".mnemotree/chromadb")
memory_core = MemoryCore(store=store)

# Store a new memory with analysis
memory = await memory_core.remember(
    content="The user mentioned they prefer coding in Python because of its readability.",
    memory_type=MemoryType.EPISODIC,
    tags=["preferences", "programming"],
    analyze=True  # Enable automatic content analysis
)

# Retrieve similar memories
related_memories = await memory_core.recall(
    "What programming languages does the user like?",
    limit=5
)

# Analyze patterns across memories
insights = await memory_core.reflect(
    min_importance=0.7  # Only consider important memories
)
```

### Option Objects (optional)

```python
from mnemotree import RecallFilters, RecallOptions, RememberOptions

memory = await memory_core.remember(
    content="Met Alex to discuss the roadmap.",
    options=RememberOptions(tags=["meeting"], source="notes")
)

memories = await memory_core.recall(
    "roadmap discussion",
    options=RecallOptions(
        limit=5,
        filters=RecallFilters(tags=["meeting"], min_importance=0.3),
    ),
)
```

### Lite Mode (CPU, no LLM)

```python
from mnemotree import MemoryCore
from mnemotree.store import ChromaMemoryStore

store = ChromaMemoryStore(persist_directory=".mnemotree/chromadb")
memory_core = MemoryCore(store=store, mode="lite")

memory = await memory_core.remember(
    content="Quick capture: met Alex to discuss the roadmap.",
    tags=["meeting"]
)
```

Set `MNEMOTREE_LITE_EMBEDDING_MODEL` to override the local embedding model.
Lite mode uses spaCy for NER/keywords; install the model with `python -m spacy download en_core_web_sm`.
To use YAKE keywords, pass `keyword_extractor=YakeKeywordExtractor()` and install `mnemotree[keywords]`.

**Alternative NER backends (optional)**

You can swap out the NER engine by passing `ner=...` into `MemoryCore`:

```python
from mnemotree import MemoryCore
from mnemotree.ner import TransformersNER

memory_core = MemoryCore(store=store, mode="lite", ner=TransformersNER(model="dslim/distilbert-NER"))
```

Extras:
- Transformers: `mnemotree[ner_hf]` (e.g. `dslim/distilbert-NER`)
- GLiNER: `mnemotree[ner_gliner]`
- Stanza: `mnemotree[ner_stanza]`

### MCP Server Configuration

The MCP server uses lite mode (local CPU embeddings) with ChromaDB by default.

**Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOTREE_MCP_PERSIST_DIR` | `.mnemotree/chromadb` | ChromaDB storage directory |
| `MNEMOTREE_MCP_COLLECTION` | `memories` | ChromaDB collection name |
| `MNEMOTREE_MCP_CHROMA_HOST` | — | Remote ChromaDB host (optional) |
| `MNEMOTREE_MCP_CHROMA_PORT` | — | Remote ChromaDB port (optional) |
| `MNEMOTREE_MCP_CHROMA_SSL` | `false` | Use SSL for remote ChromaDB |
| `MNEMOTREE_MCP_ENABLE_NER` | `false` | Enable named entity recognition |
| `MNEMOTREE_MCP_ENABLE_KEYWORDS` | `false` | Enable keyword extraction |
| `MNEMOTREE_MCP_NER_BACKEND` | — | NER backend: `spacy`, `transformers`, `gliner`, `stanza` |
| `MNEMOTREE_MCP_NER_MODEL` | — | Backend-specific model ID/path |

**Concurrency Note**

Avoid running multiple MCP processes against the same local Chroma persistence directory.
For shared memory across clients, either:
- Run a single HTTP MCP server and connect all clients to it
- Use unique `MNEMOTREE_MCP_PERSIST_DIR` values per server
- Point to a remote ChromaDB server

**Docker (optional)**

```bash
# Build and run directly
docker build -f docker/mcp/Dockerfile -t mnemotree-mcp .
docker run --rm -p 8000:8000 -v "$PWD/.mnemotree:/data" mnemotree-mcp

# Or use Compose
docker compose -f docker/mcp/docker-compose.yml up --build
```

### Advanced Configuration

```python
from mnemotree import MemoryCore
from mnemotree.core.scoring import MemoryScoring
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure custom scoring
scoring = MemoryScoring(
    importance_weight=0.4,
    recency_weight=0.4,
    relevance_weight=0.2
)

# Initialize with custom components
memory_core = MemoryCore(
    store=store,
    llm=ChatOpenAI(model_name="gpt-4"),
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    memory_scoring=scoring,
    default_importance=0.5
)
```

## 🔧 Storage Options

### ChromaDB Setup
```python
from mnemotree.store import ChromaMemoryStore

# Local persistence
store = ChromaMemoryStore(
    persist_directory=".mnemotree/chromadb"
)

# Remote ChromaDB
store = ChromaMemoryStore(
    host="localhost",
    port=8000,
    ssl=False
)
```

### Neo4j Setup
```python
from mnemotree.store import Neo4jMemoryStore

store = Neo4jMemoryStore(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="password"
)
```

## 🐳 Docker Setup

### Neo4j Setup
```bash
# Navigate to Neo4j docker compose directory
cd docker/neo4j

# Start Neo4j container
docker-compose up -d

# Default credentials:
# URL: neo4j://localhost:7687
# Username: neo4j
# Password: password
```

### ChromaDB Setup
```bash
# Navigate to ChromaDB docker compose directory
cd docker/chromadb

# Start ChromaDB container
docker-compose up -d

# Default settings:
# URL: http://localhost:8000
```

### Using Multiple Databases
You can run all required databases simultaneously:

```bash
# Start all services
docker-compose -f docker/neo4j/docker-compose.yml -f docker/chromadb/docker-compose.yml up -d

# Stop all services
docker-compose -f docker/neo4j/docker-compose.yml -f docker/chromadb/docker-compose.yml down
```

[Previous content remains the same until the Installation section...]

## 📦 Installation

Since the package is not yet available on PyPI, you'll need to install it from the local source:

```bash
# Clone the repository
git clone https://github.com/kurcontko/mnemotree.git
cd mnemotree

# Create a virtual environment
uv venv .venv

# Basic installation
uv pip install -e .

# With specific database support
uv pip install -e ".[neo4j]"     # Neo4j support
uv pip install -e ".[chroma]"    # ChromaDB support
uv pip install -e ".[sqlite_vec]"# SQLite + sqlite-vec support
uv pip install -e ".[vectors]"   # All vector database support
uv pip install -e ".[lite]"      # Local CPU embeddings (Lite mode)
uv pip install -e ".[keywords]"  # YAKE keyword extraction
uv pip install -e ".[ui]"        # UI components

# With specific LLM provider support
uv pip install -e ".[google]"     # Google AI support
uv pip install -e ".[anthropic]"  # Anthropic support
uv pip install -e ".[aws]"        # AWS support
uv pip install -e ".[huggingface]"# HuggingFace support

# Install all optional dependencies
uv pip install -e ".[all]"
```

## Roadmap

### Features in Development
- More advanced relationships
- Learnable scoring weights
- Memory compression
- Chunking large memory entries
- Enhanced relationship modeling (especially for vector stores)
- Advanced clustering algorithms
- Memory migration utilities

## 💡 Examples

Mnemotree offers a flexible and powerful framework for building LLM agents with advanced memory capabilities. Here are some examples to get you started:

### 1. LangChain Agent with Memory

This example demonstrates how to integrate Mnemotree with a LangChain agent, allowing it to store, recall, and utilize memories to enhance its responses.

**Code:** [`examples/langchain_agent.py`](examples/langchain_agent.py)

**To Run:**

1. Make sure you have the required dependencies installed (see Installation section).
2. Set your OpenAI API key as an environment variable: `OPENAI_API_KEY=your_api_key`.
3. Run the script: `uv run python examples/langchain_agent.py`

### 2. Streamlit Chat Application with Memory

This example demonstrates a Streamlit-based chat application that utilizes Mnemotree to maintain a persistent memory of the conversation.

**Code:** [`examples/memory_chat/app.py`](examples/memory_chat/app.py)

**To Run:**

1. Install with UI extras: `uv pip install -e ".[ui]"`
2. Make sure you have the required database (Neo4j or ChromaDB) running and configured correctly.
3. Set your OpenAI API key: `OPENAI_API_KEY=your_api_key`
4. Run the Streamlit app: `uv run streamlit run examples/memory_chat/app.py`

These examples demonstrate the core functionality of Mnemotree and how it can be integrated into different types of applications. You can adapt and extend these examples to build more complex and sophisticated LLM agents with advanced memory capabilities.

## Development

- Run checks: `make lint typecheck test`
- Pre-commit hooks: `make precommit-install` (uses `pre-commit.yaml`)
- CI template: `ci/github-actions-ci.yml` (copy into your CI system)

## 🤝 Contributing

We welcome contributions from the community! If you'd like to contribute, please:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write clear and concise code with proper documentation.
4. Write unit tests to ensure code quality :)
5. Submit a pull request with a detailed description of your changes.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
