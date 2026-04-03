# Production-Grade Improvement Plan

## Executive Summary

Based on analysis, the mnemotree codebase has a solid foundation but has three critical areas for production-grade improvement:

| Priority | Area | Current State | Impact |
|----------|------|--------------|--------|
| **P0** | Architectural Consolidation | Dual retriever implementations (`HybridFusionRetriever` vs `HybridRetriever`) | Code maintainability, testing burden |
| **P1** | Observability & Tracing | No instrumentation in pipeline | Debugging/latency analysis impossible |
| **P2** | Resilience & Caching | `CachedEmbeddings` exists but **not integrated** into `MemoryCore` | Redundant API calls, cost/latency |

---

## 1. Architectural Consolidation (P0)

### Problem Analysis

- `src/mnemotree/core/retrieval.py` contains `HybridFusionRetriever` (489 lines) - tightly coupled to `MemoryCore`, creates its own candidates directly
- `src/mnemotree/core/hybrid_retrieval.py` contains `HybridRetriever` (310 lines) - cleaner design with `RetrievalResult` dataclass for score provenance
- `MemoryCore._build_retriever()` (memory.py lines 1286-1309) hardcodes the legacy `HybridFusionRetriever`

**Current Usage of HybridRetriever** (broader than initially assessed):
| Location | Usage Pattern |
|----------|---------------|
| `configs.py` | `MemorySystemConfig.build_memory_system()` creates standalone `HybridRetriever` |
| `ConfiguredMemorySystem` | Holds `retriever: HybridRetriever \| None` as separate component |
| `examples/advanced_memory_demo.py` | Direct instantiation |
| `examples/quick_start_advanced.py` | Direct instantiation |
| `tests/core/test_retrieval_pipeline.py` | Unit tests |

**Key Insight**: Two parallel retrieval paths exist:
1. `MemoryCore.recall()` → uses internal `HybridFusionRetriever` (legacy)
2. `ConfiguredMemorySystem.retriever` → uses external `HybridRetriever` (modern, but not wired to MemoryCore)

### Solution: Unify on Existing `Retriever` Protocol + DI

The `Retriever` protocol already exists in `retrieval.py` (line 37-45). **No new protocol will be created.** Both retriever implementations must conform to this existing interface.

```python
# Existing protocol in core/retrieval.py - DO NOT DUPLICATE
@runtime_checkable
class Retriever(Protocol):
    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]: ...
```

```
Phase 1: Align HybridRetriever to Existing Retriever Protocol
├── Add `recall()` method to HybridRetriever matching EXISTING protocol signature
├── Internal `retrieve()` becomes implementation detail
├── Verify with: `isinstance(hybrid_retriever, Retriever)` → True
└── NO new RetrieverProtocol class - use existing `Retriever`

Phase 2: Refactor MemoryCore for DI
├── Accept optional `retriever: Retriever` in constructor
├── Create `RetrieverFactory` for default construction
├── Update `_build_retriever()` to return either implementation
└── Deprecate direct `HybridFusionRetriever` construction

Phase 3: Update configs.py Integration
├── `ConfiguredMemorySystem` injects retriever INTO MemoryCore
├── Remove separate `retriever` field (or keep for advanced composition)
└── Ensure `build_memory_system()` wires retriever correctly

Phase 4: Deprecate & Remove Legacy
├── Mark `HybridFusionRetriever` as deprecated (warnings)
├── Migration period: 2 minor versions
└── Delete `HybridFusionRetriever` in next major version

Phase 5: Expose RetrievalResult
├── Optional `recall_with_provenance()` method on MemoryCore
├── Same filtering/scoring/update_access semantics as `recall()` (or explicitly documented as read-only)
└── Returns `list[RetrievalResult]` for debugging/observability
```

### Files to Modify

| File | Changes |
|------|---------|
| `core/memory.py` | Add DI for retriever, update constructor signature |
| `core/hybrid_retrieval.py` | Add `recall()` method matching existing `Retriever` protocol |
| `core/retrieval.py` | Mark `HybridFusionRetriever` as deprecated |
| `core/__init__.py` | Export `RetrieverFactory` |
| `configs.py` | Update `MemorySystemConfig.build_memory_system()` to inject retriever into MemoryCore |
| `configs.py` | Update `ConfiguredMemorySystem` to remove/deprecate standalone `retriever` field |

**Note on `configs.py` scope**: The `build_memory_system()` method currently creates a standalone `HybridRetriever` that is NOT wired into `MemoryCore.recall()`. This is the primary integration point that must be updated to ensure DI is actually used. All code paths constructing retrievers must be audited:

```python
# Current (broken): retriever exists but isn't used by MemoryCore
memory_core = MemoryCore(store=store, llm=llm, embeddings=embeddings)
retriever = HybridRetriever(...)  # Standalone, not connected

# Target (fixed): retriever injected into MemoryCore
retriever = RetrieverFactory.create_hybrid(...)
memory_core = MemoryCore(store=store, retriever=retriever, ...)
```

### Deliverables

- [ ] `HybridRetriever.recall()` method matching `Retriever` protocol
- [ ] `RetrieverFactory` class with `create_hybrid()`, `create_basic()` methods
- [ ] `MemoryCore` accepts optional `retriever: Retriever` parameter
- [ ] `configs.py` updated to wire retriever into MemoryCore
- [ ] Deprecation warnings on `HybridFusionRetriever`
- [ ] Migration guide in docs
- [ ] Tests updated to use new interface

---

## 2. Observability & Tracing (P1)

### Problem Analysis

- Pipeline stages (Vector → NER → BM25 → RRF → Rerank) have no timing or score visibility
- Only `logger.debug()` calls with basic duration logging
- Users cannot diagnose why specific memories rank higher/lower
- No integration with standard observability tools

### Solution: OpenTelemetry Instrumentation

```
Tracing Points:
├── recall() entry/exit (total latency)
├── _retrieve_vector_candidates() (embedding + vector search)
├── _retrieve_entity_candidates() (NER extraction + entity query)
├── BM25 search (index_manager.search)
├── rrf_fuse() (fusion scoring)
├── reranker.rerank() (cross-encoder latency)
└── _update_access() (metadata updates)

Metrics to Capture:
├── Latency per stage (histogram)
├── Candidate counts per stage (gauge)
├── Score distributions (histogram)
└── Reranker invocation rate (counter)

Note: Cache hit rate metrics are deferred to P2 (Caching Integration).
Placeholder metric stubs may be added in P1 but will emit zeros until P2.

### Metrics Label Cardinality Policy

**CRITICAL**: Unbounded labels cause metrics explosion and performance degradation.

**Allowed Labels (bounded cardinality):**
| Label | Allowed Values | Cardinality |
|-------|---------------|-------------|
| `stage_name` | `vector`, `entity`, `bm25`, `fusion`, `rerank` | 5 |
| `retrieval_mode` | `basic`, `hybrid` | 2 |
| `cache_status` | `hit`, `miss` | 2 |
| `reranker_type` | `none`, `flashrank`, `cross_encoder` | 3 |

**FORBIDDEN Labels (unbounded):**
- `memory_id` - millions of unique values
- `query_hash` - unbounded
- `user_id` - unbounded without tenant isolation
- `tag_name` - user-defined, unbounded

**Implementation:**
```python
# Metrics with bounded labels only
recall_latency = Histogram(
    "mnemotree_recall_latency_seconds",
    "Recall operation latency",
    labelnames=["stage_name", "retrieval_mode"],  # Bounded!
)

# Score stats as histogram buckets, NOT labels
score_distribution = Histogram(
    "mnemotree_score_distribution",
    "Score distribution across results",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
```
```

### Data Privacy & PII Policy

**CRITICAL**: Trace data must not contain user content to avoid PII exposure.

```
Allowed in Spans:
├── memory_id (UUIDs only)
├── candidate_count, stage_name, latency_ms
├── score_stats (min, max, mean - numeric only)
├── retrieval_stage names
└── configuration flags (e.g., rerank_enabled)

FORBIDDEN in Spans:
├── memory.content (user text)
├── query text (user input)
├── entity names extracted from content
├── tags or metadata containing user data
└── embedding vectors
```

Configuration for sensitive environments:
```python
MNEMOTREE_OTEL_REDACT_QUERIES=true  # Default: true
MNEMOTREE_OTEL_INCLUDE_SCORES=true  # Safe: numeric only
MNEMOTREE_OTEL_INCLUDE_IDS=true     # Safe: UUIDs only
```

### Implementation Structure

```
src/mnemotree/
├── observability/
│   ├── __init__.py
│   ├── _noop.py         # No-op stubs when OTel not installed
│   ├── tracing.py       # OpenTelemetry tracer setup
│   ├── metrics.py       # Prometheus/OTLP metrics
│   └── decorators.py    # @traced, @timed decorators
```

### Graceful Degradation (No OTel Installed)

**CRITICAL**: Core code must not fail if `observability` extras are not installed.

**Strategy: Lazy Import + No-Op Fallback**

```python
# observability/__init__.py
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

def get_tracer(name: str):
    if not OTEL_AVAILABLE:
        from ._noop import NoOpTracer
        return NoOpTracer()
    return trace.get_tracer(name)
```

```python
# observability/_noop.py
class NoOpSpan:
    def set_attribute(self, key, value): pass
    def add_event(self, name, attributes=None): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

class NoOpTracer:
    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()
```

```python
# Usage in core/retrieval.py - SAFE regardless of extras
from mnemotree.observability import get_tracer

tracer = get_tracer(__name__)  # Returns NoOpTracer if OTel not installed

async def recall(...):
    with tracer.start_as_current_span("recall") as span:
        span.set_attribute("limit", limit)  # No-op if OTel missing
        ...
```

**Rules:**
1. Never `import opentelemetry` directly in core modules
2. Always use `mnemotree.observability` wrappers
3. `@traced` decorator must be no-op when OTel unavailable
4. No runtime errors for missing optional deps

### Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `observability` optional dependency group |
| New: `observability/tracing.py` | OTel tracer configuration, span helpers |
| New: `observability/decorators.py` | `@traced` decorator for async methods |
| `core/retrieval.py` | Instrument `recall()`, `_collect_rrf_candidates()` |
| `core/hybrid_retrieval.py` | Instrument `retrieve()`, `_fuse_candidates()` |
| `rerankers/base.py` | Instrument `rerank()` |

### Configuration

```python
# Optional: Auto-configure via environment
MNEMOTREE_OTEL_ENABLED=true
MNEMOTREE_OTEL_ENDPOINT=http://localhost:4317
MNEMOTREE_OTEL_SERVICE_NAME=mnemotree
```

### Deliverables

- [ ] `observability` extras group in pyproject.toml
- [ ] `@traced` decorator with automatic span attributes
- [ ] No-op fallback module (`_noop.py`) for graceful degradation
- [ ] Lazy import pattern - core code never imports OTel directly
- [ ] Span attributes include: `candidate_count`, `stage_name`, `score_stats` (numeric only)
- [ ] PII redaction enforced by default (no query text, no content in spans)
- [ ] Span status set on failures and exceptions recorded in spans
- [ ] Metrics with bounded label cardinality (max 5 values per label)
- [ ] Optional Prometheus metrics exporter
- [ ] Placeholder stubs for cache metrics (activated in P2)
- [ ] Example Grafana dashboard JSON

---

## 3. Resilience & Caching (P2)

### Problem Analysis

- `embeddings/cache.py` exists with `CachedEmbeddings` wrapper (205 lines, well-implemented)
- **NOT integrated into `MemoryCore`** - embedding caching is opt-in manual wrapping
- `_resolve_embeddings()` in memory.py creates raw embedders without cache

### Solution: Automatic Cache Integration

```
Phase 1: Configuration
├── Add `enable_embedding_cache: bool = False` to ModeDefaultsConfig (opt-in initially)
├── Add `embedding_cache_size: int = 1000`
├── Add `embedding_cache_ttl: float = 3600`

Phase 2: Auto-Wrapping
├── _resolve_embeddings() wraps with CachedEmbeddings when enabled
└── Expose cache stats via MemoryCore.embedding_cache_stats

Phase 3: Persistence (Optional)
├── Add SQLite-backed cache for cross-session persistence
├── Default location: `~/.cache/mnemotree/embeddings.db` (XDG-compliant)
├── Configurable via `MNEMOTREE_CACHE_DIR` environment variable
└── Implement warm-up from stored embeddings
```

### Cache Key Schema

**Key composition (avoid collisions):**

```
key = hash(
    embedder_id + embedder_config_hash + tenant_scope + text
)
```

- `embedder_id`: model name or provider identifier
- `embedder_config_hash`: hash of embedder settings (dim, normalize, etc.)
- `tenant_scope`: user/tenant id or "global" if none
- `text`: raw input text (hashed only; never stored as plaintext in cache key)

### Persistent Cache Specification

**Default Storage Location (XDG Base Directory Spec):**
```python
import os
from pathlib import Path

def get_cache_dir() -> Path:
    """Return cache directory following XDG spec."""
    if custom := os.getenv("MNEMOTREE_CACHE_DIR"):
        return Path(custom)

    # XDG_CACHE_HOME or ~/.cache
    xdg_cache = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(xdg_cache) / "mnemotree"

# Default: ~/.cache/mnemotree/embeddings.db
```

**Multi-Process Safety:**

| Approach | Safety | Performance | Recommended For |
|----------|--------|-------------|------------------|
| SQLite WAL mode | ✅ Safe | Good (concurrent reads) | Most deployments |
| File locking | ✅ Safe | Poor (serialized) | Fallback only |
| Process-local + periodic sync | ⚠️ Eventual | Best | High-throughput |

**Implementation (SQLite WAL):**
```python
# embeddings/persistent_cache.py
import sqlite3
from contextlib import contextmanager

class PersistentEmbeddingCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Multi-process safe
            conn.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s for locks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON embeddings(expires_at)")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
```

**Disk Usage Limits:**
```python
MNEMOTREE_CACHE_MAX_SIZE_MB=500  # Default: 500MB
MNEMOTREE_CACHE_EVICTION_POLICY=lru  # lru | ttl | size
```

### Concurrency & Multi-Process Considerations

**In-Memory Cache Limitations:**
- `CachedEmbeddings` uses `asyncio.Lock` - safe for single-process async
- **NOT shared across processes** (e.g., gunicorn workers, multiprocessing)
- Each process maintains independent cache → potential redundant embeddings

**Guidance for Production Deployments:**

| Deployment | Recommendation |
|------------|---------------|
| Single process (uvicorn, MCP server) | Enable in-memory cache |
| Multi-worker (gunicorn, k8s pods) | Use persistent cache (P3) or external cache |
| Serverless (Lambda, Cloud Run) | Disable or use external Redis/Memcached |

**Opt-Out Configuration:**
```python
# Disable caching entirely
MemoryCore(
    store=store,
    mode_defaults=ModeDefaultsConfig(enable_embedding_cache=False),
)

# Or via environment
MNEMOTREE_EMBEDDING_CACHE_ENABLED=false
```

**Memory Pressure Mitigation:**
- Default `max_size=1000` limits memory to ~50-100MB (typical embeddings)
- TTL eviction prevents unbounded growth
- Monitor via `MemoryCore.embedding_cache_stats`

### Files to Modify

| File | Changes |
|------|---------|
| `core/memory.py` | Add cache config, wrap embeddings in `_resolve_embeddings()` |
| `embeddings/cache.py` | Add optional persistence layer |
| New: `embeddings/persistent_cache.py` | SQLite-backed embedding cache |

### Deliverables

- [ ] `CachingConfig` dataclass with cache parameters
- [ ] Caching **opt-in** initially (`enable_embedding_cache=False` default)
- [ ] `MemoryCore.embedding_cache_stats` property
- [ ] Environment variable overrides (`MNEMOTREE_EMBEDDING_CACHE_*`)
- [ ] Documentation on multi-process limitations
- [ ] Cache key schema includes model + config hash + tenant scope (no plaintext text)
- [ ] Persistent cache with XDG-compliant default location (`~/.cache/mnemotree/`)
- [ ] SQLite WAL mode for multi-process safety
- [ ] Configurable disk usage limits (`MNEMOTREE_CACHE_MAX_SIZE_MB`)
- [ ] Activate cache hit rate metrics from P1 stubs

---

## Implementation Roadmap

```
Week 1-2: Architectural Consolidation
├── Day 1-2: Add recall() to HybridRetriever, verify protocol compliance
├── Day 3-4: Design RetrieverFactory interface
├── Day 5-6: Implement DI in MemoryCore constructor
├── Day 7-8: Update configs.py to wire retriever into MemoryCore
├── Day 9: Add deprecation warnings to HybridFusionRetriever
├── Day 10: Update all tests, write migration guide
├── Add feature flag/escape hatch to force legacy retriever for rollback

Week 3-4: Observability
├── Day 1-2: Create observability module structure
├── Day 3-4: Implement @traced decorator with OTel + PII redaction
├── Day 5-7: Instrument all pipeline stages
├── Day 8: Add placeholder cache metric stubs
├── Day 9: Create example dashboard + docs
├── Day 10: Security review of trace attributes

Week 5: Caching Integration
├── Day 1-2: Add CachingConfig to ModeDefaultsConfig (opt-in)
├── Day 3: Integrate into _resolve_embeddings() with env overrides
├── Day 4: Activate cache hit rate metrics (connect P1 stubs)
├── Day 5: Document multi-process limitations + opt-out guidance
```

### Dependencies Between Phases

```
P0 (Consolidation) ──────────────────────────────────┐
                                                     │
P1 (Observability) ──► Cache metric stubs ───────────┤
                                                     ▼
P2 (Caching) ──────► Activates cache metrics ◄───────┘
```

- P1 ships with stub/placeholder for `embedding_cache_hits` and `embedding_cache_misses`
- P2 activates these metrics when `CachedEmbeddings` is integrated
- P0 and P1 can proceed in parallel; P2 depends on P1 metric infrastructure

---

## Testing Strategy

| Component | Test Type | Coverage Target |
|-----------|-----------|----------------|
| RetrieverFactory | Unit | 100% |
| DI in MemoryCore | Integration | All retriever variants |
| Tracing decorators | Unit + Mock | Span creation, attributes |
| Cache integration | Unit + Performance | Hit rates, eviction |

---

## Breaking Changes

**Policy Alignment:** All deprecations/removals must follow `docs/API.md` stability policy (minimum 2 minor releases before removal).

| Change | Migration Path |
|--------|---------------|
| `_build_retriever()` deprecated | Use `retriever=` constructor param |
| `HybridFusionRetriever` deprecated | Use `HybridRetriever` via factory (warning for 2 versions) |
| `ConfiguredMemorySystem.retriever` wiring | Retriever now injected into MemoryCore internally |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PII in traces | Redaction enabled by default, security review in Week 4 |
| Cache memory pressure | Opt-in initially, configurable limits, monitoring via stats |
| Multi-process cache misses | Document limitations, provide persistent cache option |
| Breaking `configs.py` consumers | Maintain backward compat for `retriever` field during deprecation |
| Rollout regressions | Feature flags/escape hatches for retriever + cache, staged rollout |

---

## Appendix: Current Architecture Reference

### Retriever Protocol (existing)

```python
# src/mnemotree/core/retrieval.py
@runtime_checkable
class Retriever(Protocol):
    async def recall(
        self,
        query: str | MemoryQuery | MemoryQueryBuilder,
        limit: int | None,
        scoring: bool,
        update_access: bool,
    ) -> list[MemoryItem]: ...
```

### RetrievalResult (modern, underutilized)

```python
# src/mnemotree/core/hybrid_retrieval.py
@dataclass
class RetrievalResult:
    memory: MemoryItem
    scores: dict[str, float]  # Score per stage
    final_score: float
    retrieval_stages: list[RetrievalStage]
    metadata: dict[str, Any] | None = None
```

### CachedEmbeddings (existing, not integrated)

```python
# src/mnemotree/embeddings/cache.py
class CachedEmbeddings(Embeddings):
    def __init__(
        self,
        embedder: Embeddings,
        *,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
    ) -> None: ...
```
