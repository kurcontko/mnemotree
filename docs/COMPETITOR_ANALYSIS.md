# Competitor Code Analysis: Hindsight, EverMemOS, SimpleMem

**Date**: 2026-03-15
**Purpose**: Code-level analysis of top-performing agent memory systems to inform
mnemotree development priorities.

---

## Hindsight — 91.4% LongMemEval, 89.6% LoCoMo

**Repo**: https://github.com/vectorize-io/hindsight
**Stack**: PostgreSQL + pgvector, Python, FastAPI

### Architecture: 4-Network Memory

Hindsight organizes all memory into exactly 4 types stored in one `MemoryUnit`
table with a `fact_type` constraint:

| Network | fact_type | Purpose | Confidence |
|---------|-----------|---------|------------|
| World | `world` | Objective external facts | NULL |
| Experience | `experience` | Agent's own activities | NULL |
| Opinion | `opinion` | Beliefs with confidence | 0.0-1.0 (required) |
| Observation | `observation` | Entity summaries from consolidation | 0.0-1.0 |

**MemoryUnit fields** (models.py):
- `id`, `bank_id` (tenant), `text`, `context`
- `embedding` (Vector 1024-dim, pgvector HNSW)
- `occurred_start`, `occurred_end` (temporal range)
- `mentioned_at` (when mentioned in conversation)
- `fact_type` (world/experience/opinion/observation)
- `confidence_score` (float 0-1, required for opinion/observation)
- `metadata` (JSONB), `tags` (visibility scope)

### Retain: LLM Fact Extraction

Pipeline in `engine/retain/orchestrator.py`:

1. Extract facts via LLM with verbose what/when/where/who/why format
2. Each fact produces: `fact_type`, `occurred_start/end`, `entities`, `causal_relations`
3. Embed with 1024-dim model
4. Store + create entity links and causal links

LLM prompts force complete information capture — "COMPLETE details, never
summarize" — with explicit coreference resolution.

### Recall: 4-Channel Retrieval with RRF Fusion

`engine/search/retrieval.py` runs 4 parallel channels:

**Channel 1 — Semantic**: pgvector HNSW cosine similarity, over-fetches 5x
(min 100), threshold >= 0.3

**Channel 2 — BM25**: PostgreSQL full-text search via configurable backend
(vchord, pg_textsearch, or native ts_rank_cd)

**Channel 3 — Graph (Spreading Activation)**: BFS from semantic entry points,
activation decays at 0.8 per hop, causal links boosted 2x, stops at min
activation 0.1 or budget

**Channel 4 — Temporal**: Two-phase entry via date indexes then similarity join,
temporal proximity scoring: `1.0 - min(days_from_mid / (total_days / 2), 1.0)`

**Fusion**: Reciprocal Rank Fusion (k=60):
```
score(d) = sum_over_lists(1 / (k + rank(d)))
```

### Reflect: Hierarchical Retrieval Agent

`engine/reflect/agent.py` implements a multi-iteration agent (max 10) with tools:

1. Check mental models first (user-curated, highest quality)
2. Check observations (auto-consolidated)
3. Fall back to raw facts (mandatory if higher levels return 0)

Tools: `list_observations`, `get_observation`, `recall`, `expand`, `done`

### Confidence Tracking: Trend-Based (Not Direct Update)

`engine/reflect/observations.py` computes dynamic trends from evidence
timestamps:

```
recent_density = count(evidence in last 30 days) / 30
older_density = count(evidence in 30-90 days) / 60
ratio = recent_density / older_density

ratio > 1.5 → STRENGTHENING
ratio < 0.5 → WEAKENING
else → STABLE
no recent evidence → STALE
all evidence recent → NEW
```

Confidence score is initialized at 1.0 for opinions and not explicitly updated —
trend is computed dynamically.

### Key Takeaway for MnemoTree

The main performance driver is the **4-channel RRF fusion** retrieval. MnemoTree
has all 4 components (vector search, BM25, graph traversal, temporal filtering)
but they are not fused via RRF in the agent recall path. Wiring this could be the
single highest-impact change.

---

## EverMemOS — 93.05% LoCoMo (Highest Reported)

**Repo**: https://github.com/EverMind-AI/EverMemOS
**Stack**: MongoDB + Milvus + Elasticsearch, Python

### Architecture: 3-Phase Workflow

#### Phase 1: Episodic Trace Formation (MemCell Extraction)

`stage1_memcells_extraction.py` performs boundary detection on conversations:

- Uses LLM to detect topic boundaries in conversation flow
- Extracts contiguous segments into `MemCell` objects
- Immediately computes downstream memories (Episode, Foresight, EventLog) before
  moving to next segment

**MemCell fields**:
- `user_id_list`, `original_data` (normalized messages)
- `timestamp`, `summary`
- `episode` (narrative), `foresight` (prospective facts)
- `event_log` (atomic facts), `keywords`

#### Phase 2: Semantic Consolidation (Adaptive Clustering)

`cluster_manager/manager.py` implements online incremental clustering:

```
new_centroid = (old_centroid * count + new_vector) / (count + 1)
```

- Assigns each MemCell to best cluster if cosine similarity >= threshold AND
  time gap <= max_time_gap_seconds
- Creates new cluster if no match
- No explicit compaction — centroid averaging IS the consolidation

**ProfileManager** extracts semantic profiles (skills, personality, habits) from
clusters with incremental merging.

#### Phase 3: Reconstructive Recollection (Agentic Multi-Round)

`stage3_memory_retrivel.py` is the core innovation driving +19.7% multi-hop:

**Round 1** (always runs):
```
Hybrid Search (Top 20) → Rerank (Top 10) → LLM Sufficiency Judge
```

**Round 2** (if insufficient):
```
Generate 3 Refined Queries → Parallel Hybrid Search × 3 → Multi-RRF Fusion → Merge with Round 1 → Rerank Top 20
```

Multi-RRF voting: documents appearing in multiple query results rank higher.
This is what enables multi-hop — each query targets a different aspect of the
answer.

### Storage: Tiered Backend

- **MongoDB**: Primary persistence (MemCells, EventLogs, Profiles)
- **Milvus**: Vector similarity search
- **Elasticsearch**: BM25 keyword search + full-text
- **In-memory**: ClusterState for fast centroid operations

### Key Takeaway for MnemoTree

The **multi-round agentic retrieval with sufficiency checking** is the key
differentiator. When initial retrieval is insufficient, generating targeted
follow-up queries and fusing results via multi-RRF dramatically improves
multi-hop accuracy. This is a retrieval strategy, not a storage change.

---

## SimpleMem — 43.24% F1, ~550 Tokens/Query

**Repo**: https://github.com/aiming-lab/SimpleMem
**Stack**: Tantivy (Rust FTS), local embeddings, Python

### Architecture: 3-Stage Pipeline

#### Stage 1: Semantic Structured Compression

`core/memory_builder.py` processes dialogues in sliding windows (size 40,
overlap 2):

- LLM extracts `MemoryEntry` objects with forced coreference resolution (no
  pronouns) and absolute timestamps (ISO 8601)
- Parallel window processing via ThreadPoolExecutor (16 workers)

#### Stage 2: Online Semantic Synthesis

Not a separate process — during extraction, previous 3 entries are passed as
context to the LLM to avoid generating duplicates. Deduplication happens at
write time, not post-hoc.

#### Stage 3: Intent-Aware Retrieval Planning

`core/hybrid_retriever.py` implements the core innovation:

1. **Analyze information requirements**: LLM infers query type (factual/
   temporal/relational/explanatory) and extracts key entities
2. **Generate targeted queries**: 1-4 minimal queries, each targeting a specific
   information requirement
3. **Parallel multi-view retrieval**:
   - Semantic: Top 25 (dense embedding search)
   - Lexical: Top 5 (BM25 via Tantivy)
   - Symbolic: Top 5 (metadata filters on persons/entities/timestamps)
4. **ID-based merge and deduplication**

### MemoryEntry: Multi-View Indexed Unit

```
entry_id: UUID
lossless_restatement: str   # Self-contained, no pronouns
keywords: List[str]         # BM25 terms
timestamp: Optional[str]    # ISO 8601
location: Optional[str]
persons: List[str]
entities: List[str]
topic: Optional[str]
```

Three index layers:
- **s_k (Semantic)**: 1024-dim dense embeddings
- **l_k (Lexical)**: Tantivy FTS on lossless_restatement
- **r_k (Symbolic)**: SQL metadata filters

### Intent-to-Retrieval Mapping

| Query Type | Strategy |
|-----------|----------|
| Factual | Semantic + Lexical (direct lookup) |
| Temporal | Symbolic (timestamp filter) + Semantic |
| Relational | Semantic + Lexical + Symbolic (persons/entities) |
| Explanatory | Multi-view with expansion queries |

### Key Takeaway for MnemoTree

MnemoTree already implements SimpleMem's intent-aware filtering (via
`enable_intent_filter` and `IntentClassifier`). The main gap is the
**multi-view retrieval** combining semantic + lexical + symbolic in a single
query. The `lossless_restatement` concept (no pronouns, absolute timestamps)
could improve memory quality during ingestion.

---

## Comparative Architecture Summary

| Feature | Hindsight | EverMemOS | SimpleMem | MnemoTree |
|---------|-----------|-----------|-----------|-----------|
| **LoCoMo** | 89.6% | 93.05% | 43.24% F1 | 70.0% |
| **Storage** | PostgreSQL + pgvector | MongoDB + Milvus + ES | Tantivy + local | SQLite + sqlite-vec |
| **Retrieval channels** | 4 (semantic, BM25, graph, temporal) | 3 (semantic, BM25, rerank) | 3 (semantic, lexical, symbolic) | 3 (semantic, BM25, graph) |
| **Fusion** | RRF (k=60) | Multi-RRF (multi-query) | ID-based merge | RRF (partial) |
| **Multi-hop** | Graph spreading activation | Agentic multi-round | Query expansion | PPR graph retrieval |
| **Confidence** | Trend-based (dynamic) | N/A | N/A | ObservationStatus enum |
| **Compaction** | Observations via reflect | Online centroid update | Write-time dedup | Manual compact_summary |
| **Memory types** | 4 networks | MemCell + Episode + EventLog | MemoryEntry | MemoryType enum (8) |
| **Coreference** | LLM extraction | LLM extraction | Forced no-pronouns | Heuristic normalizer |
| **Agent retrieval** | 10-iteration tool loop | 2-round sufficiency check | 2-round reflection | Single pass |

---

## Priority Improvements for MnemoTree

Based on this analysis, ranked by expected impact on LoCoMo:

### 1. Multi-Channel RRF Fusion in Agent Recall (HIGH IMPACT)

Wire the existing 4 retrieval channels (vector, BM25, graph, temporal) into a
unified RRF fusion pipeline for `agent_recall_with_summary`. Hindsight proves
this is the core differentiator at the 89%+ level.

### 2. Multi-Round Agentic Retrieval (HIGH IMPACT)

Add sufficiency checking after initial retrieval. If the top-k results don't
answer the query, generate 2-3 refined follow-up queries and merge via
multi-RRF. EverMemOS proves this drives +19.7% on multi-hop.

### 3. Wire Confidence Score into Observation Semantics (MEDIUM IMPACT)

The existing `confidence` field on MemoryItem should be coupled with
ObservationStatus. Use Hindsight's trend-based approach: compute confidence
dynamically from evidence density over recent vs older time windows.

### 4. Automatic Background Compaction (MEDIUM IMPACT)

Replace manual `agent_compact_summary` with automatic compaction triggered after
N observations or on a timer. Use EverMemOS's incremental centroid approach for
efficiency.

### 5. Lossless Restatement at Ingestion (MEDIUM IMPACT)

Apply SimpleMem's forced coreference resolution and absolute timestamp
normalization during memory ingestion. MnemoTree already has a heuristic
normalizer — extending it with LLM-backed resolution would improve retrieval
quality.

### 6. Temporal Proximity Scoring in Retrieval (LOW-MEDIUM)

Add Hindsight's temporal channel as a 4th retrieval signal: score memories by
proximity to a query's time reference, separate from general recency decay.
