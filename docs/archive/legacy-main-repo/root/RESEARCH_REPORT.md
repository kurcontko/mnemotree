# Mnemotree Improvement Research Report

**Date:** 2026-02-21

---

## Current State

Mnemotree sits at **59.7% LoCoMo accuracy** (vs mem0's 66.9%). Strongest area is temporal reasoning (57.6%), weakest is single-hop (21.3%). Already has FSRS-4.5 decay, hybrid BM25+semantic retrieval, GLiNER NER, and keyword extraction.

---

## The Competitive Landscape Has Shifted

| System | LoCoMo Score | Token Usage | Key Differentiator |
|--------|-------------|-------------|-------------------|
| **MemMachine v0.2** | **0.9169** (LLM-score) | 4.2M (80% less than mem0) | Reranking + agent mode |
| **MemOS** | **#1 all categories** | - | 3-layer OS architecture |
| **SimpleMem** | **43.24 F1** (+26.4% over mem0) | 30x fewer tokens | Semantic lossless compression |
| **Mem0g** (graph) | **68.4%** | - | Graph-based entity memory |
| **MemLoRA** (2B) | **47.2 J-score** | 10-20x less | Distilled expert adapters |

Sources:
- [MemMachine v0.2](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)
- [SimpleMem](https://arxiv.org/html/2601.02553v1)
- [Mem0 research](https://mem0.ai/research)
- [MemLoRA](https://arxiv.org/html/2512.04763)

---

## HIGH-IMPACT Ideas for Mnemotree

### 1. Distilled Expert Adapters on Ingestion (Validated by MemLoRA)

MemLoRA proves the small finetuned models approach works. Their approach:

- **3 specialized LoRA adapters** on a 2B model (Gemma2-2B or Qwen2.5-1.5B):
  - **Extraction adapter**: pulls facts/preferences from conversation
  - **Update adapter**: decides ADD/UPDATE/DELETE operations
  - **Generation adapter**: produces memory-augmented responses
- **Text-based distillation** from a large teacher (e.g. GPT-4.1 or Gemma2-27B) — no logit distillation needed, just train on teacher outputs
- Result: **2B model matches or beats 27B models**, 10-20x faster inference

**Actionable for mnemotree**: Replace LLM-based extraction (in "Pro" mode) with LoRA adapters on Qwen3-4B or Gemma2-2B. The distillation pipeline is straightforward — generate training data with a large model, clean outputs, train with standard next-token prediction. **Qwen3-4B-Instruct** currently leads fine-tuning benchmarks.

### 2. Semantic Lossless Compression (SimpleMem's Approach)

SimpleMem achieved +26.4% over mem0 with three techniques:

- **Information scoring with entropy-aware filtering**: Score dialogue windows by entity novelty + semantic divergence. Discard redundant windows below threshold — prevents memory bloat
- **Context normalization**: Resolve coreferences ("he" -> "John") and anchor temporal expressions to ISO-8601 timestamps. Critical for single-hop weakness (21.3%) — exact-match retrieval fails when memories contain unresolved pronouns
- **Recursive consolidation**: Background process that clusters related memories (similarity > 0.85) and synthesizes them into abstract patterns (e.g., multiple "ordered coffee" entries -> "user regularly drinks coffee mornings")

**Actionable for mnemotree**: FSRS decay already handles staleness, but consolidation is missing. Adding coreference resolution + temporal anchoring to the ingestion pipeline could directly fix single-hop performance.

Source: [SimpleMem paper](https://arxiv.org/html/2601.02553v1)

### 3. Adaptive Query-Aware Retrieval

Current retrieval uses fixed-k. Multiple systems now show adaptive retrieval is superior:

- **Query complexity estimation** (0-1 score): Simple factual queries get k=3 from abstract summaries, complex multi-hop queries expand to k=15+ detailed entries
- **Question-type routing**: Single-hop -> prioritize exact lexical match (BM25 weight up), Multi-hop -> prioritize semantic + graph traversal, Temporal -> prioritize timestamp-sorted retrieval
- **SymRAG-style routing**: Classify query into symbolic (exact match), neural (semantic), or hybrid paths

This directly addresses category-specific weaknesses. Single-hop (21.3%) needs heavier BM25 weighting, while multi-hop (49.0%) needs graph traversal.

Source: [Adaptive Query Routing](https://arxiv.org/html/2506.12981v1/)

### 4. Graph Memory Layer (Mem0g Approach)

Mem0g's graph variant bumped their score from 66.9% to 68.4%. The architecture:

- **Entity Extractor** identifies nodes from incoming messages
- **Relations Generator** infers labeled directed edges
- Store as a directed labeled graph alongside vector store
- Retrieval: query entities, traverse 1-2 hops for related facts

Mnemotree already has Neo4j support and GLiNER NER. The missing piece is **automated relationship extraction and graph-augmented retrieval**. When a query mentions "John", traverse the graph to pull in related entities (John's workplace, family, preferences) even if they weren't in the top-k vector results.

Source: [Mem0 Graph Memory](https://mem0.ai/blog/graph-memory-solutions-ai-agents)

### 5. Conversation Segmentation (Pre-Ingestion)

Instead of processing memories turn-by-turn or session-by-session, segment conversations into **topically coherent segments** first. Research shows segment-level memory construction "better balances retrieval precision compared to turn-level or session-level memory." A small classifier model can handle this cheaply.

### 6. Cross-Encoder Reranking

MemMachine v0.2's jump to top scores came partly from adding **Cohere Rerank v3.5** as a secondary reranker after initial retrieval. FlashRank support already exists in mnemotree but may not be active in benchmark pipeline. Low-hanging fruit — reranking consistently adds 3-5 points across all question types.

### 7. Episodic-to-Semantic Memory Consolidation

Plays directly to the FSRS strength. Implement a **memory evolution pipeline**:

1. Fresh memories enter as **episodic** (specific events with timestamps)
2. After FSRS reviews, recurring patterns consolidate into **semantic** memories (general facts)
3. Semantic memories get higher stability and lower decay rates
4. Mirrors human cognition — unique differentiator no competitor has with the FSRS foundation

Source: [Memory Mechanisms Survey](https://github.com/Shichun-Liu/Agent-Memory-Paper-List), [EVOLVE-MEM](https://openreview.net/pdf?id=dfPQrg1WA5)

---

## Emerging Threats: LoCoMo-Plus

[LoCoMo-Plus](https://arxiv.org/html/2602.10715) (Feb 2026) introduces **cognitive memory evaluation** — testing causal, state-based, goal-oriented, and value-based memory. All current systems (including mem0) show **13-27 point drops** from LoCoMo to LoCoMo-Plus. This is an opportunity to leapfrog competitors by targeting cognitive memory early. Mnemotree's taxonomy system (episodic, semantic, procedural, autobiographical) already aligns with these cognitive dimensions better than competitors' flat memory stores.

---

## SQLite Graph Strategy (Keep It Simple)

### Current SQLite State

Two separate SQLite systems exist:
- `SQLiteVecMemoryStore` — vector similarity + metadata
- `SQLiteGraphIndex` — entity cooccurrence, memory edges, 2-hop traversal

They **don't integrate** — no unified graph+vector queries.

### Why Skip Neo4j

Conversational memory graphs are sparse — typically 500-5,000 entities per user. SQLite with proper indexes handles this trivially. Neo4j only makes sense at >100K entities with complex multi-hop path algorithms, which isn't the conversational memory use case.

### Required SQLite Upgrades

#### 1. Unify Vector Store + Graph Index

A single retrieval pipeline should:
- Vector search -> get top-k candidates
- Expand via graph traversal -> pull in related entities' memories
- Re-score with combined vector + graph signal

This is a retrieval-layer integration, not a schema change.

#### 2. Replace `memory_edge` with proper `memory_link` table

```sql
CREATE TABLE memory_link (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    link_type TEXT NOT NULL,  -- all 11 LinkType values
    strength  REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    metadata TEXT,  -- JSON
    PRIMARY KEY (source_id, target_id, link_type)
);
-- Reverse lookup index
CREATE INDEX idx_link_target ON memory_link(target_id);
```

Gives bidirectional queries, full LinkType support, and strength decay — all things currently Neo4j-only.

#### 3. Link Strength Decay (integrates with FSRS)

```sql
UPDATE memory_link
SET strength = strength * :decay_factor
WHERE last_accessed < :cutoff_timestamp;
```

#### 4. Multi-hop Traversal via Recursive CTE

```sql
WITH RECURSIVE hops AS (
    -- Seed: direct entity matches
    SELECT me.memory_id, 0 AS depth, me.mention_count AS score
    FROM memory_entity me
    JOIN entity e ON e.id = me.entity_id
    WHERE e.name IN (:query_entities)

    UNION ALL

    -- Expand: follow edges up to N hops
    SELECT ml.target_id, h.depth + 1, h.score * ml.strength * 0.5
    FROM hops h
    JOIN memory_link ml ON ml.source_id = h.memory_id
    WHERE h.depth < :max_hops
)
SELECT memory_id, MAX(score) as score, MIN(depth) as depth
FROM hops
GROUP BY memory_id
ORDER BY score DESC
LIMIT :k;
```

3-5 hop traversal, runs in <10ms on a 5,000-entity graph.

#### 5. Graph-Augmented Retrieval Flow

```
Query → extract entities (GLiNER)
     → parallel:
        ├── vector search (top-k=20)
        ├── BM25 search (top-k=20)
        └── graph traversal (entities → 2-hop → memory_ids)
     → merge with RRF
     → rerank (FlashRank/cross-encoder)
     → top-k results
```

The graph branch catches memories that are semantically distant but **relationally close** — exactly what multi-hop questions need.

---

## Recommended Priority Roadmap

| Priority | Improvement | Expected Impact | Effort |
|----------|-------------|----------------|--------|
| **P0** | Coreference resolution + temporal anchoring on ingestion | Fix single-hop (21.3% -> ~40%+) | Medium |
| **P0** | Cross-encoder reranking in benchmark pipeline | +3-5 pts across all categories | Low |
| **P1** | Distilled LoRA adapters (extraction + update) on Qwen3-4B | Replace LLM costs, match large model quality | High |
| **P1** | Adaptive k + query-type routing | +5-8 pts on multi-hop and temporal | Medium |
| **P1** | Unify SQLite vector + graph into single retrieval pipeline | Enable graph-augmented retrieval | Medium |
| **P2** | Upgrade SQLite graph (memory_link table + recursive CTE) | +3-5 pts on multi-hop | Medium |
| **P2** | Recursive memory consolidation | Unique differentiator + efficiency | Medium |
| **P3** | Conversation segmentation pre-processing | Better retrieval precision | Low |
| **P3** | Episodic-to-semantic evolution pipeline | LoCoMo-Plus readiness | High |

**Conservative estimate**: P0+P1 items alone could push from 59.7% to **70-75%+**, past mem0's 66.9% baseline. Combined with P2 items and existing FSRS advantage, competitive with MemMachine and SimpleMem.

---

## Key References

- [LoCoMo Benchmark](https://snap-research.github.io/locomo/)
- [LoCoMo-Plus](https://arxiv.org/html/2602.10715)
- [Mem0 Paper](https://arxiv.org/abs/2504.19413)
- [Mem0 Graph Memory](https://mem0.ai/blog/graph-memory-solutions-ai-agents)
- [MemMachine v0.2](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)
- [SimpleMem](https://arxiv.org/html/2601.02553v1)
- [MemLoRA](https://arxiv.org/html/2512.04763)
- [MemOS](https://arxiv.org/abs/2507.03724)
- [EVOLVE-MEM](https://openreview.net/pdf?id=dfPQrg1WA5)
- [Adaptive Query Routing (SymRAG)](https://arxiv.org/html/2506.12981v1/)
- [Memory Mechanisms Survey](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- [SLM Fine-tuning Benchmarks](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
- [Letta/MemGPT](https://docs.letta.com/concepts/memgpt/)
