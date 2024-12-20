# 🌳 Mnemotree

An Advanced Memory Management and Retrieval System for LLM Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

<p align="center">
  <img src="assets/mnemotree-final.png" alt="Mnemotree Logo" width="300">
</p>

## Overview

Mnemotree is a framework that enhances Large Language Model (LLM) agents with biologically-inspired memory capabilities. It provides a unified system for storing, retrieving, and analyzing structured knowledge representations, designed to seamlessly integrate with popular LLM frameworks like LangChain and Autogen.

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
  - Neo4j support for graph-based relationships
  - Extensible architecture for custom storage implementations
- **Advanced Querying**:
  - Semantic similarity search
  - Complex filtering
  - Relationship-based queries
  - Importance-based filtering

## 🚀 Getting Started

### Prerequisites

```bash
- Python 3.10+
- pip or Poetry for dependency management
- Access to an LLM API (e.g., OpenAI)
- Vector store (ChromaDB) or graph database (Neo4j) for storage
```

### Installation

```bash
pip install mnemotree
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

### Advanced Configuration

```python
from mnemotree import MemoryCore
from mnemotree.core.scoring import MemoryScoring, DecayFunction
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure custom scoring
scoring = MemoryScoring(
    importance_weight=0.3,
    recency_weight=0.2,
    emotion_weight=0.2,
    decay_function=DecayFunction.POWER_LAW
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

# Basic installation
pip install -e .

# With specific database support
pip install -e ".[neo4j]"     # Neo4j support
pip install -e ".[chroma]"    # ChromaDB support
pip install -e ".[vectors]"   # All vector database support
pip install -e ".[ui]"        # UI components

# With specific LLM provider support
pip install -e ".[google]"     # Google AI support
pip install -e ".[anthropic]"  # Anthropic support
pip install -e ".[aws]"        # AWS support
pip install -e ".[huggingface]"# HuggingFace support

# Install all optional dependencies
pip install -e ".[all]"
```

or 

```bash
pip install -r requirements.txt
```

## Roadmap

### Database Integrations
- ✅ Neo4j
- ✅ ChromaDB
- 🚧 PostgreSQL (pgvector) - In Progress
- 🚧 Milvus - In Progress
- 🚧 Qdrant - In Progress
- 🚧 Azure Cosmos DB - Planned
- 🚧 Pinecone - Planned
- 🚧 Weaviate - Planned

### Framework Integrations
- ✅ LangChain
- 🚧 Autogen - In Progress
- 🚧 Llama-Index - Planned
- 🚧 Semantic Kernel - Planned

### LLM Provider Support
Framework is tightly coupled with langchain framework so you can use any langchain compatible llms and embeddings for example:
- OpenAI
- Google AI
- Anthropic 
- AWS Bedrock 
- HuggingFace 

It might be changed in future

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
3. Run the script: `python examples/langchain_agent.py`

### 2. Streamlit Chat Application with Memory

This example demonstrates a Streamlit-based chat application that utilizes Mnemotree to maintain a persistent memory of the conversation.

**Code:** [`examples/memory_chat/app.py`](examples/memory_chat/app.py)

**To Run:**

1. Install Streamlit: `pip install streamlit`
2. Make sure you have the required database (Neo4j or ChromaDB) running and configured correctly.
3. Set your OpenAI API key: `OPENAI_API_KEY=your_api_key`
4. Run the Streamlit app: `streamlit run examples/memory_chat/app.py`

These examples demonstrate the core functionality of Mnemotree and how it can be integrated into different types of applications. You can adapt and extend these examples to build more complex and sophisticated LLM agents with advanced memory capabilities.

## 🤝 Contributing

We welcome contributions from the community! If you'd like to contribute, please:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write clear and concise code with proper documentation.
4. Write unit tests to ensure code quality :)
5. Submit a pull request with a detailed description of your changes.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
