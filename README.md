# 🌳 Mnemotree

An Advanced Memory Management and Retrieval System for LLM Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

<p align="center">
  <img src="assets/mnemotree-logo.png" alt="Mnemotree Logo" width="300">
</p>

## Overview

Mnemotree is a framework that enhances Large Language Model (LLM) agents with biologically-inspired memory capabilities. It provides a unified system for storing, retrieving, and analyzing structured knowledge representations, designed to seamlessly integrate with popular LLM frameworks like LangChain and Autogen.

It includes a ready-to-use **Model Context Protocol (MCP)** server, allowing easy integration with Claude Desktop, Codex CLI, and other MCP-compliant tools with a single command.

## API Stability

The project has a small set of stable public entry points and a larger set of experimental internals.
See docs/API.md for what is considered public API, what is experimental, and what compatibility guarantees are provided.

## ⚡ MCP Quickstart (FastMCP)

Run the MCP server directly from the project source using `uvx`:

```bash
# Run via stdio (for Claude Desktop etc)
uvx --from . --with mnemotree[mcp_server] mnemotree-mcp
```

Or connect via HTTP (for multiple clients):

```bash
uvx --from . --with mnemotree[mcp_server] mnemotree-mcp run --transport http --port 8000
```

### Usage from other tools (Claude Desktop, etc.)

To use this MCP server from other applications while developing locally, reference the absolute path to your repository.

**Claude Desktop Config (`claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "mnemotree": {
      "command": "uvx",
      "args": [
        "--from",
        "/absolute/path/to/mnemotree",
        "--with",
        "mnemotree[mcp_server]",
        "mnemotree-mcp"
      ],
      "env": {
        "MNEMOTREE_MCP_PERSIST_DIR": "/absolute/path/to/mnemotree/.mnemotree/chromadb"
      }
    }
  }
}
```

**Command Line (from any directory):**

```bash
uvx --from /path/to/mnemotree --with mnemotree[mcp_server] mnemotree-mcp
```

### Usage directly from GitHub

You can also run the server without cloning the repository locally:

```bash
uvx --from "git+https://github.com/kurcontko/mnemotree.git" --with "mnemotree[mcp_server]" mnemotree-mcp
```

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
- Flair: `mnemotree[ner_flair]`
- Stanza: `mnemotree[ner_stanza]`

### MCP Server (FastMCP)

Mnemotree ships with a lightweight MCP server wrapper around `MemoryCore` (lite + Chroma).

**Install MCP + lite + Chroma extras**

```bash
uv pip install -e ".[mcp,lite,chroma]"
```

**Run via uvx (stdio, local)**

Use this when a client launches the server process itself (stdio transport).

```bash
uvx --from . --with mnemotree[mcp_server] mnemotree-mcp
```

**Run as a separate process (HTTP, shared)**

Use this when multiple Codex instances should share one memory store.

```bash
uvx --from . --with mnemotree[mcp_server] mnemotree-mcp run --transport http --port 8000
```

Then connect MCP clients to `http://localhost:8000/mcp`.

**Concurrency note**

For shared memory, prefer a single MCP server process and have all clients connect to it.
Avoid running multiple MCP processes against the same local Chroma persistence directory.
If you need multiple MCP servers, set unique `MNEMOTREE_MCP_PERSIST_DIR` values per server,
or point them at a remote Chroma server.

**Environment variables**

- `MNEMOTREE_MCP_PERSIST_DIR` (default: `.mnemotree/chromadb`)
- `MNEMOTREE_MCP_COLLECTION` (default: `memories`)
- `MNEMOTREE_MCP_CHROMA_HOST`, `MNEMOTREE_MCP_CHROMA_PORT`, `MNEMOTREE_MCP_CHROMA_SSL`
- `MNEMOTREE_MCP_ENABLE_NER`, `MNEMOTREE_MCP_ENABLE_KEYWORDS` (both default to `false`)
- `MNEMOTREE_MCP_NER_BACKEND` (e.g. `spacy`, `transformers`, `gliner`, `stanza`, `flair`)
- `MNEMOTREE_MCP_NER_MODEL` (backend-specific model id/path)

**Docker (optional)**

There is a lightweight Docker setup under `docker/mcp`.

Build and run directly:

```bash
docker build -f docker/mcp/Dockerfile -t mnemotree-mcp .
docker run --rm -p 8000:8000 -v "$PWD/.mnemotree:/data" mnemotree-mcp
```

Or use Compose:

```bash
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
