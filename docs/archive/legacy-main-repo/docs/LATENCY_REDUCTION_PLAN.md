# Mnemotree Latency Reduction Plan

## Executive Summary

This document outlines strategies to reduce latency in `remember` and `recall` operations without sacrificing retrieval quality. Based on analysis of the codebase, the main latency contributors are:

1. **Embedding generation** - ML model inference for each operation
2. **NER/Entity extraction** - SpaCy/GLiNER/LLM inference
3. **Reranking** - Cross-encoder inference (recall only)
4. **Storage I/O** - Database operations (ChromaDB, SQLite-vec, Neo4j, Milvus)
5. **BM25 lexical search** - Index operations
6. **Analysis/Summarization** - LLM calls (remember only)

---

## Current Architecture Analysis

### Remember Pipeline (`_remember_sync`)
```
Content → Enrichment Pipeline → Memory Construction → Persist
           ├── Embedding (async)
           ├── NER (async, optional)
           ├── Keywords (async, optional)
           ├── Summarization (async, optional)
           └── Analysis (async, optional)
```

**Current optimizations:**
- Tasks run in parallel via `asyncio.gather`
- Async ingestion queue available (`async_ingest=True`)

### Recall Pipeline (`HybridFusionRetriever.recall`)
```
Query → Candidate Collection → RRF Fusion → Scoring → Reranking → Results
         ├── Vector search (async)
         ├── Entity search (async)
         ├── BM25 search (sync)
         └── Keywords (async, optional)
```

**Current optimizations:**
- Vector/entity/keyword tasks run in parallel
- BM25 has in-memory index with memory cache
- Optional reranking stage

---

## Phase 1: Quick Wins (Low Effort, High Impact)

### 1.1 Embedding Caching Layer
**Impact: 30-50% reduction for repeated queries**
**Effort: Low**

Add an LRU cache for embedding results to avoid redundant model inference.

```python
# src/mnemotree/core/_internal/embedding_cache.py
from functools import lru_cache
from typing import Callable, Awaitable
import hashlib

class EmbeddingCache:
    def __init__(self, embedder, maxsize: int = 1000):
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._maxsize = maxsize

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def embed_query(self, text: str) -> list[float]:
        key = self._hash_text(text)
        if key in self._cache:
            return self._cache[key]
        embedding = await self._embedder.aembed_query(text)
        if len(self._cache) >= self._maxsize:
            # Remove oldest entry (simple FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = embedding
        return embedding
```

**Implementation locations:**
- [core/_internal/enrichment.py](../src/mnemotree/core/_internal/enrichment.py) - `StandardEnrichmentPipeline._get_embedding`
- [core/retrieval.py](../src/mnemotree/core/retrieval.py) - `BaseRetriever._get_embedding`

### 1.2 Lazy Model Loading Optimization
**Impact: 100-500ms reduction on first call**
**Effort: Low**

Models (SpaCy, sentence-transformers, cross-encoders) are lazily loaded. Add pre-warming option.

```python
# Add to MemoryCore
async def warmup(self) -> None:
    """Pre-load all ML models to avoid cold-start latency."""
    # Warm up embedder
    await self.embedder.aembed_query("warmup")
    # Warm up NER
    if self.ner:
        await self.ner.extract_entities("warmup")
    # Warm up reranker
    if hasattr(self, 'reranker') and self.reranker:
        self.reranker._load_model()
```

### 1.3 Reduce Default Candidate Pool Size
**Impact: 10-20% reduction in recall latency**
**Effort: Low**

Current: `candidate_k = min(max(50, resolved_limit * 5), ...)`

Optimize based on actual needs:
```python
# Dynamic candidate sizing based on collection size
if cache_len < 100:
    candidate_k = min(cache_len, resolved_limit * 3)
elif cache_len < 1000:
    candidate_k = min(50, resolved_limit * 4)
else:
    candidate_k = min(100, resolved_limit * 5)
```

**Location:** [core/retrieval.py#L344-347](../src/mnemotree/core/retrieval.py)

---

## Phase 2: Medium-Term Optimizations (Medium Effort)

### 2.1 Batch Embedding Support
**Impact: 40-60% reduction for batch operations**
**Effort: Medium**

Current `batch_remember` calls `remember` sequentially. Add true batching:

```python
async def batch_remember_optimized(
    self,
    contents: list[str],
    *,
    analyze: bool = False,
    context: dict[str, Any] | None = None,
) -> list[MemoryItem]:
    # Batch embed all contents at once
    embeddings = await self.embedder.aembed_documents(contents)

    # Batch NER extraction
    if self.ner:
        ner_tasks = [self.ner.extract_entities(c) for c in contents]
        ner_results = await asyncio.gather(*ner_tasks)

    # Create memories with pre-computed embeddings
    memories = []
    for i, content in enumerate(contents):
        memory_data = self._build_memory_data(
            content=content,
            embedding=embeddings[i],  # Use pre-computed
            # ...
        )
        memories.append(MemoryItem(**memory_data))

    # Batch persist
    await asyncio.gather(*[self.persistence.save(m) for m in memories])
    return memories
```

**Location:** [core/memory.py#L941-965](../src/mnemotree/core/memory.py)

### 2.2 Async BM25 Index Operations
**Impact: 5-15% reduction in recall latency**
**Effort: Medium**

Current BM25 search is synchronous. Move to async:

```python
async def search_async(self, query: str, k: int) -> list[tuple[str, float]]:
    return await asyncio.to_thread(self.search, query, k)
```

**Location:** [core/_internal/indexing.py#L334-358](../src/mnemotree/core/_internal/indexing.py)

### 2.3 Lighter NER Models for Speed-First Mode
**Impact: 50-80% reduction in NER latency**
**Effort: Medium**

Current SpaCy default: `en_core_web_sm` (~20ms/call)
Faster alternative: `en_core_web_trf` is slower; use smaller models or skip NER for simple queries.

Add NER mode configuration:
```python
@dataclass
class NerConfig:
    enabled: bool = True
    ner_type: str = "spacy"
    model: str = "en_core_web_sm"
    mode: str = "balanced"  # "fast", "balanced", "quality"

    @property
    def effective_model(self) -> str:
        if self.mode == "fast":
            return "en_core_web_sm"  # Smallest
        elif self.mode == "quality":
            return "en_core_web_lg"  # Most accurate
        return self.model
```

### 2.4 Skip Unnecessary Enrichment Steps
**Impact: 20-40% reduction in remember latency**
**Effort: Low**

Add granular control to skip expensive steps when not needed:

```python
@dataclass
class EnrichmentFlags:
    compute_embedding: bool = True  # Required
    extract_entities: bool = True
    extract_keywords: bool = True
    analyze_content: bool = False  # Off by default
    summarize_content: bool = False  # Off by default
```

**Location:** [core/_internal/enrichment.py#L53-63](../src/mnemotree/core/_internal/enrichment.py)

---

## Phase 3: Advanced Optimizations (Higher Effort)

### 3.1 Approximate Nearest Neighbor (ANN) Index Tuning
**Impact: 10-30% reduction for large datasets**
**Effort: Medium-High**

#### ChromaDB
```python
# Current: Uses default HNSW settings
# Optimization: Tune for speed
collection = client.create_collection(
    name="memories",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,  # Lower for faster indexing
        "hnsw:search_ef": 50,  # Lower for faster search (tradeoff with recall)
        "hnsw:M": 16,  # Connections per node
    }
)
```

#### SQLite-vec
```python
# Add IVF index configuration
VECTOR_INDEX_CONFIG = {
    "n_lists": 100,  # Number of clusters
    "n_probes": 10,  # Clusters to search
}
```

#### Milvus
```python
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",  # or "IVF_SQ8" for compression
    "params": {"nlist": 128}
}
search_params = {"nprobe": 16}  # Trade accuracy for speed
```

### 3.2 Tiered Storage Strategy
**Impact: Variable, depends on access patterns**
**Effort: High**

Implement hot/warm/cold memory tiers:

```python
class TieredMemoryStore:
    def __init__(self):
        self.hot_cache: dict[str, MemoryItem] = {}  # In-memory, <1ms
        self.warm_store: ChromaDBStore  # Local SSD, ~5ms
        self.cold_store: Neo4jStore  # Network, ~20ms

    async def get_similar_memories(self, query_embedding, top_k):
        # Try hot cache first
        hot_results = self._search_hot(query_embedding, top_k)
        if len(hot_results) >= top_k:
            return hot_results

        # Fall back to warm/cold
        remaining = top_k - len(hot_results)
        warm_results = await self.warm_store.get_similar_memories(
            query_embedding, remaining
        )
        return self._merge_results(hot_results, warm_results)
```

### 3.3 Query-Adaptive Pipeline
**Impact: 20-40% average latency reduction**
**Effort: Medium**

Skip unnecessary retrieval stages based on query characteristics:

```python
async def adaptive_recall(self, query: str, limit: int) -> list[MemoryItem]:
    query_length = len(query.split())
    has_named_entities = self._quick_entity_check(query)

    # Short, simple queries: vector-only
    if query_length < 5 and not has_named_entities:
        return await self._vector_only_recall(query, limit)

    # Entity-heavy queries: prioritize entity search
    if has_named_entities:
        return await self._entity_boosted_recall(query, limit)

    # Default: full hybrid
    return await self._full_hybrid_recall(query, limit)

def _quick_entity_check(self, query: str) -> bool:
    # Fast heuristic: check for capitalized words
    words = query.split()
    return any(w[0].isupper() for w in words if w)
```

### 3.4 Faster Reranker Options
**Impact: 50-80% reduction in reranking latency**
**Effort: Low-Medium**

Current options ranked by speed:
1. **No reranker** - 0ms (baseline)
2. **FlashRank TinyBERT** (~2ms/query) - Already available ✓
3. **Cross-encoder MiniLM** (~10ms/query) - Current default
4. **Cross-encoder Large** (~50ms/query) - Highest quality

Add dynamic reranker selection:

```python
class AdaptiveReranker(BaseReranker):
    def __init__(self, fast: BaseReranker, quality: BaseReranker):
        self.fast = fast  # FlashRank
        self.quality = quality  # CrossEncoder

    async def rerank(self, query: str, candidates: list[MemoryItem], top_k: int):
        # Use fast reranker for many candidates
        if len(candidates) > 20:
            return await self.fast.rerank(query, candidates, top_k)
        # Use quality reranker for final refinement
        return await self.quality.rerank(query, candidates, top_k)
```

---

## Phase 4: Infrastructure & Architecture

### 4.1 Connection Pooling
**Impact: 10-30% reduction for concurrent operations**
**Effort: Medium**

Add connection pooling for database backends:

```python
# Neo4j with connection pool
from neo4j import AsyncGraphDatabase

driver = AsyncGraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=50,
    connection_acquisition_timeout=30.0,
)

# Milvus with connection pool
from pymilvus import connections
connections.connect(
    pool_size=10,
    max_retry=3,
)
```

### 4.2 Parallel Store Queries
**Impact: Up to 50% reduction for multi-stage recall**
**Effort: Low**

Already partially implemented; ensure all stages run concurrently:

```python
# Current implementation (good)
vector_task = asyncio.create_task(self._retrieve_vector_candidates(...))
entity_task = asyncio.create_task(self._retrieve_entity_candidates(...))
(vector, _), entities = await asyncio.gather(vector_task, entity_task)

# Enhancement: Add BM25 to parallel group
bm25_task = asyncio.create_task(asyncio.to_thread(
    self.index_manager.search, query, candidate_k
))
results = await asyncio.gather(vector_task, entity_task, bm25_task)
```

### 4.3 Background Index Updates
**Impact: 20-30% reduction in remember latency**
**Effort: Medium**

Move index updates to background:

```python
class BackgroundIndexer:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self._queue: asyncio.Queue[MemoryItem] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def schedule_update(self, memory: MemoryItem) -> None:
        self._queue.put_nowait(memory)

    async def _worker(self) -> None:
        while True:
            memory = await self._queue.get()
            self.index_manager.add(memory)
            self._queue.task_done()
```

---

## Configuration Profiles

Add pre-configured profiles for different use cases:

```python
# src/mnemotree/profiles.py

PROFILE_SPEED = {
    "retrieval_config": {
        "enable_bm25": False,  # Skip lexical search
        "enable_prf": False,
        "rerank_candidates": 0,  # Skip reranking
    },
    "ner_config": {
        "enabled": False,  # Skip NER
    },
    "ingestion_config": {
        "async_ingest": True,
    },
}

PROFILE_BALANCED = {
    "retrieval_config": {
        "enable_bm25": True,
        "enable_prf": False,
        "rerank_candidates": 20,
        "reranker_backend": "flashrank",  # Fast reranker
    },
    "ner_config": {
        "enabled": True,
        "ner_type": "spacy",
        "model": "en_core_web_sm",
    },
}

PROFILE_QUALITY = {
    "retrieval_config": {
        "enable_bm25": True,
        "enable_prf": True,
        "prf_docs": 5,
        "prf_terms": 8,
        "rerank_candidates": 50,
        "reranker_backend": "cross_encoder",
    },
    "ner_config": {
        "enabled": True,
        "ner_type": "gliner",  # or "llm" for best quality
    },
}
```

---

## Metrics & Monitoring

Add latency tracking to identify bottlenecks:

```python
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class OperationMetrics:
    total_ms: float
    embedding_ms: float = 0
    ner_ms: float = 0
    store_ms: float = 0
    bm25_ms: float = 0
    rerank_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "breakdown": {
                "embedding": self.embedding_ms,
                "ner": self.ner_ms,
                "store": self.store_ms,
                "bm25": self.bm25_ms,
                "rerank": self.rerank_ms,
            }
        }

# Usage
async def recall_with_metrics(self, query: str, limit: int) -> tuple[list[MemoryItem], OperationMetrics]:
    metrics = OperationMetrics(total_ms=0)
    started = time.perf_counter()

    # Track embedding
    embed_start = time.perf_counter()
    query_embedding = await self._get_embedding(query)
    metrics.embedding_ms = (time.perf_counter() - embed_start) * 1000

    # ... rest of recall

    metrics.total_ms = (time.perf_counter() - started) * 1000
    return results, metrics
```

---

## Implementation Priority

| Priority | Optimization | Effort | Impact | Dependencies |
|----------|-------------|--------|--------|--------------|
| 1 | Embedding caching | Low | High | None |
| 2 | Model pre-warming | Low | Medium | None |
| 3 | Reduce candidate pool | Low | Medium | None |
| 4 | Configuration profiles | Low | High | None |
| 5 | Batch embedding | Medium | High | None |
| 6 | Async BM25 | Medium | Low | None |
| 7 | Skip unnecessary enrichment | Low | Medium | None |
| 8 | Lighter NER modes | Medium | Medium | NER refactor |
| 9 | ANN index tuning | Medium | Medium | Store-specific |
| 10 | Adaptive reranker | Medium | Medium | Multiple rerankers |
| 11 | Query-adaptive pipeline | Medium | Medium | Heuristics tuning |
| 12 | Background indexing | Medium | Medium | Queue infrastructure |
| 13 | Tiered storage | High | High | Architecture change |

---

## Expected Results

Based on current benchmark data (45 queries, 44 memories):

| Configuration | Current P50 | Target P50 | Improvement |
|--------------|-------------|------------|-------------|
| Full pipeline | ~25ms | ~12ms | 52% |
| Balanced | ~15ms | ~8ms | 47% |
| Speed-first | ~10ms | ~4ms | 60% |

### Quality Preservation

Key metrics to monitor:
- **Precision@1**: Maintain >0.90
- **Recall@5**: Maintain >0.85
- **NDCG@10**: Maintain >0.92
- **MRR**: Maintain >0.94

Run ablation studies with each optimization to verify quality preservation.

---

## Next Steps

1. **Benchmark current state** - Establish baselines with detailed timing breakdowns
2. **Implement Phase 1** - Quick wins for immediate impact
3. **A/B test configurations** - Validate quality preservation
4. **Iterate on Phase 2-3** - Based on profiling results
5. **Document trade-offs** - Create user-facing configuration guide
