# Mnemotree vs Competitors

## Why Choose Mnemotree?

| Feature | Mnemotree | mem0 | Memary | A-Mem | MemOS |
|---------|-----------|------|---------|-------|-------|
| **LoCoMo Benchmark** | ✅ 70.0% partial credit | 66.9% | — | — | — |
| **FSRS-4.5 Decay** | ✅ Scientifically-validated power-law forgetting | ❌ | ❌ | ❌ | ❌ |
| **MCP Protocol** | ✅ Native MCP server | ❌ | ❌ | ❌ | ❌ |
| **Hybrid Retrieval** | ✅ Semantic + BM25 + entity + RRF + reranking | ⚠️ Basic | ⚠️ Basic | ❌ | ❌ |
| **Multi-Store Backend** | ✅ Neo4j, SQLite, ChromaDB | ⚠️ Limited | ⚠️ Limited | ❌ | ❌ |
| **Memory Type Taxonomy** | ✅ 9 types + categories | ⚠️ Basic | ⚠️ Basic | ✅ | ❌ |
| **Graph Relationships** | ✅ Neo4j + SQLite graphs | ❌ | ⚠️ Basic | ✅ Zettelkasten | ❌ |
| **Lite Mode (CPU-only)** | ✅ No API costs (~65MB install) | ❌ | ❌ | ❌ | ❌ |
| **Framework Agnostic** | ✅ PydanticAI core + optional adapters | ✅ | ⚠️ | ❌ | ❌ |
| **Open Source** | ✅ MIT | ✅ | ✅ | ✅ | ✅ |

## Unique Advantages

### 1. Science-Based Forgetting Curves

Mnemotree is the **only** agent memory system implementing FSRS-4.5 (Free Spaced Repetition Scheduler), a scientifically-validated model used by millions via Anki.

**How it works:**
```python
R(t, S) = (1 + factor * t / S) ^ (-decay_power)
```

Where:
- `R(t, S)` = retrievability at time `t` with stability `S`
- Per-type stability (semantic: 30 days, episodic: 7 days, working: 1 hour)
- Automatic importance decay based on actual retrieval success

**Why it matters:** Other systems use arbitrary decay functions or no decay at all, leading to memory bloat and poor retrieval precision.

### 2. MCP Protocol Support

Integrate with **any** MCP-compliant client (Claude Desktop, Cline, Codex) via a single config line:

```bash
uvx --from "git+https://github.com/kurcontko/mnemotree.git" mnemotree-mcp
```

No other memory framework offers this level of tool integration.

### 3. Multi-Backend Flexibility

Choose the right storage for your use case:
- **ChromaDB**: Fast prototyping, local-first
- **Neo4j**: Complex relationship queries, knowledge graphs
- **SQLite+vec**: Embedded, zero-config, single-file

Competitors lock you into a single backend.

### 4. True Lite Mode

Run entirely on CPU with local embeddings (all-MiniLM-L6-v2) - **zero API costs, zero cloud dependencies**. Perfect for:
- Privacy-sensitive applications
- Offline environments
- Cost-conscious development

### 5. Comprehensive Memory Types

9 memory types based on cognitive science:
- **Declarative**: Episodic, Semantic, Autobiographical, Prospective
- **Non-Declarative**: Procedural, Priming, Conditioning
- **Working**: Short-term processing
- **Entities**: Extracted entity tracking

Each type has optimized decay parameters and retrieval strategies.

## Performance

From our benchmarks (see `benchmarks/results/`):

| Metric | Mnemotree (RRF+BM25) | Baseline |
|--------|----------------------|----------|
| Recall@5 | 0.92 | 0.78 |
| MRR | 0.85 | 0.71 |
| Latency (p95) | 45ms | 120ms |

## When to Choose Each

### Choose Mnemotree if:
- ✅ You need scientifically-grounded memory decay
- ✅ You want MCP protocol support
- ✅ You need complex relationship queries (graph DB)
- ✅ You want to run offline/local (Lite mode)
- ✅ You need fine-grained control over memory types

### Choose mem0 if:
- ✅ You want maximum community adoption
- ✅ You prefer managed cloud services
- ✅ You need enterprise support options

### Choose A-Mem if:
- ✅ Your primary use case is knowledge graph building
- ✅ You're focused on research applications
- ✅ You can tolerate early-stage software

### Choose Memary if:
- ✅ You specifically need Entity Knowledge Store patterns
- ✅ You want opinionated memory architecture

## Migration Guides

### From mem0

```python
# mem0
from mem0 import Memory
m = Memory()
m.add("User prefers Python", user_id="alice")
results = m.search("programming", user_id="alice")

# Mnemotree equivalent
from mnemotree import MemoryCore
from mnemotree.store import ChromaMemoryStore

store = ChromaMemoryStore()
memory = MemoryCore(store=store)
await memory.remember("User prefers Python", tags=["user:alice"])
results = await memory.recall("programming", filters={"tags": ["user:alice"]})
```

### From LangChain Memory

```python
# LangChain ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# Mnemotree + LangChain
from mnemotree import MemoryCore
from mnemotree.store import ChromaMemoryStore

store = ChromaMemoryStore()
core = MemoryCore(store=store)

# Store interaction
await core.remember(
    content=f"User: {user_input}\nAssistant: {response}",
    memory_type="episodic",
    tags=["conversation"]
)

# Retrieve for context
recent = await core.recall("", limit=10, filters={"tags": ["conversation"]})
context = "\n".join([m.to_str_llm() for m in recent])
```

## References

- [FSRS-4.5 Algorithm](https://github.com/open-spaced-repetition/fsrs4anki)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Memory in the Age of AI Agents (Survey)](https://arxiv.org/abs/2512.13564)
