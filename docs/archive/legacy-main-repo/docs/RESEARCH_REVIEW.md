# Mnemotree Memory Core: Research Review & Improvement Roadmap

## Current State Assessment

### What mnemotree does well

1. **Hybrid retrieval pipeline** -- Vector + BM25 + NER entity matching with RRF fusion is a solid architecture. The HybridFusionRetriever with weighted RRF (vector: 0.6, bm25: 0.3, entity: 0.1) is a reasonable default.

2. **Scoring formula** -- The 3-factor scoring (recency + relevance + importance) mirrors the architecture from Park et al. 2023's "Generative Agents" paper, which is the standard in AI agent memory systems.

3. **Power-law recency** -- `scoring.py` uses `(t/stability + 1)^power` for recency, which is actually ahead of many systems still using pure exponential.

4. **Enrichment pipeline** -- The lite mode with local embeddings (all-MiniLM-L6-v2), spaCy NER, and keyword extraction provides a zero-LLM-cost ingestion path.

5. **Builder pattern** -- `MemoryCoreBuilder` with fluent config keeps complexity manageable.

### Critical issues (confirmed by research)

1. **Three disconnected decay models** -- `models.py:224` (linear, broken), `scoring.py:125` (exponential, too slow), `adaptive_decay.py:328` (linear/days, disconnected). Research confirms this fragmentation is the #1 issue to fix.

2. **Exponential decay is the wrong curve** -- FSRS (now used in Anki with 15M+ users) and FadeMem both demonstrate that power-law curves empirically outperform exponential for aggregate memory decay. Your DECAY.md already identified this.

3. **`decay_and_reinforce()` is a stub** -- The adaptive decay system is completely disconnected from the core scoring pipeline. This means the entire AdaptiveImportanceSystem in experimental/ has no effect on production behavior.

4. **Hash-based novelty detection is fragile** -- Using `hash(content[:200])` misses semantic similarity entirely. Research shows embedding-based approaches (cosine similarity thresholds) are standard practice.

5. **No memory consolidation** -- Memories accumulate indefinitely with no merging, archival, or compression. Every major 2025-2026 memory framework (FadeMem, MemOS, G-Memory, SimpleMem) addresses this.

---

## Research-Backed Improvements (No LLM Required)

### Priority 1: Unified Decay Model (FSRS Power-Law)

**Research basis**: FSRS (Free Spaced Repetition Scheduler), integrated into Anki since 2023, uses a power-law forgetting curve that empirically outperforms exponential by 20-30% fewer reviews for the same retention. FadeMem (Jan 2026) extends this with a stretched exponential for AI memory specifically.

**The FSRS formula**:
```
R(t, S) = (1 + factor * t / S) ^ (-decay_power)

where:
  factor = 0.9^(-1/decay_power) - 1    # ~0.2346 when decay_power=0.5
  t = seconds since last access
  S = stability (seconds for retrievability to drop to 90%)
  decay_power = 0.5 (default, tunable per memory type)
```

With defaults F=19/81 and C=-0.5:
- After 1 day (S=7 days): R = 0.981 (1.9% decay)
- After 7 days (S=7 days): R = 0.900 (10% decay, by definition)
- After 30 days (S=7 days): R = 0.745 (25.5% decay)
- After 365 days (S=7 days): R = 0.279 (72.1% decay)

**Why it's better than current exponential**:
- Power-law has a heavier tail: memories persist longer at long timescales but decay faster initially
- This matches aggregate human forgetting curves (Kahana & Adler 2002)
- Parameterized by stability (intuitive) rather than raw rate
- Single formula replaces all three current decay models

**FadeMem extension** (for memory-type-specific behavior):
```
v(t) = v(0) * exp(-lambda * (t - tau) ^ beta)

LTM memories: beta = 0.8 (sub-linear, slow fade)
STM/working:  beta = 1.2 (super-linear, fast fade)
lambda_adaptive = lambda_base * exp(-mu * importance)
```

Half-lives: ~11 days (LTM), ~5 days (STM). Results: 82.1% critical fact retention at 55% storage.

**Cost**: Zero LLM calls. Pure math.

**Sources**:
- https://expertium.github.io/Algorithm.html
- https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm
- https://borretti.me/article/implementing-fsrs-in-100-lines
- https://arxiv.org/abs/2601.18642 (FadeMem)
- https://memory.psych.upenn.edu/files/pubs/KahaAdle02.pdf

---

### Priority 2: Memory-Type-Specific Decay Profiles

**Research basis**: Both FSRS and FadeMem demonstrate that different memory categories need different decay parameters. FSRS uses per-card difficulty-adjusted stability; FadeMem uses dual layers (SML/LML) with different beta exponents.

**Proposed mapping for mnemotree MemoryTypes**:

| MemoryType | Stability (days) | beta | Rationale |
|---|---|---|---|
| SEMANTIC | 30 | 0.8 | Facts persist; slow sub-linear decay |
| EPISODIC | 14 | 1.0 | Standard exponential-like decay |
| PROCEDURAL | 60 | 0.7 | Skills are durable; very slow decay |
| WORKING | 1 | 1.3 | Transient; fast super-linear decay |
| AUTOBIOGRAPHICAL | 21 | 0.8 | Personal facts; moderately persistent |
| PROSPECTIVE | 7 | 1.2 | Future intentions; expire faster |
| PRIMING | 3 | 1.5 | Exposure effects; rapid decay |
| CONDITIONING | 30 | 0.8 | Learned associations; persistent |

**FadeMem's hysteresis-based tier promotion** (no LLM needed):
- Promote to long-term if `importance >= 0.7`
- Demote to short-term if `importance < 0.3`
- Hysteresis gap prevents oscillation

**Cost**: Zero LLM calls. Configuration-only.

---

### Priority 3: Embedding-Based Novelty Detection

**Research basis**: Current hash-based novelty (`hash(content[:200])`) is fragile. Semantic similarity using existing embeddings is both more accurate and already infrastructure-supported in mnemotree.

**Approach**:
```python
# At remember() time, before storing:
query_embedding = await self.get_embedding(content)
similar = await self.store.get_similar_memories(
    query=content, query_embedding=query_embedding, top_k=5
)
max_similarity = max(cosine_similarity(query_embedding, m.embedding) for m in similar)

if max_similarity > 0.95:    # near-duplicate
    return NoveltyLevel.REDUNDANT
elif max_similarity > 0.85:  # very similar
    return NoveltyLevel.ROUTINE
elif max_similarity > 0.70:  # related
    return NoveltyLevel.FAMILIAR
else:
    return NoveltyLevel.NEW
```

This replaces the entire `_content_hashes` dict and survives restarts (queries the store, not in-memory state).

**Cost**: One embedding call (already computed) + one vector search (already supported). No LLM calls.

---

### Priority 4: Lightweight Reranking Upgrade

**Research basis**: Cross-encoders outperform most LLMs at reranking while being orders of magnitude cheaper. ColBERT late interaction is 100x faster than cross-encoders while maintaining quality. Current FlashRank with TinyBERT is already a good choice but there are better options.

**Current state**: `reranker_model="ms-marco-TinyBERT-L-2-v2"` via FlashRank.

**Research findings on model quality (2025 benchmarks)**:

| Model | BEIR Score | Latency (100 docs) | Notes |
|---|---|---|---|
| TinyBERT-L-2 (current) | Low | ~20ms | Fast but low quality |
| MiniLM-L-6-v2 | ~47 | ~50ms | Best speed/accuracy trade-off |
| MiniLM-L-12-v2 | ~49 | ~80ms | Best among small models |
| mxbai-rerank-base-v2 | ~54 | ~150ms | Apache 2.0, outperforms Cohere |
| BAAI/bge-reranker-base | ~53 | ~120ms | Strong multilingual |

**Recommendation**: Upgrade default to `ms-marco-MiniLM-L-6-v2` for a 2-3x quality improvement at minimal latency cost. Keep TinyBERT as the "fast" option.

**ColBERT consideration**: For larger memory stores (>10K memories), consider ColBERT-style late interaction where document representations are pre-computed at ingestion time. This gives cross-encoder-quality results at BM25-like query speeds.

**Cost**: Zero LLM calls. Small model inference only (~50ms per rerank).

**Sources**:
- https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025
- https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
- https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/

---

### Priority 5: Memory Consolidation Pipeline

**Research basis**: FadeMem, SimpleMem, G-Memory (NeurIPS 2025), and MemOS all implement memory consolidation. The pattern is consistent: cluster similar memories, merge/archive redundant ones, promote important ones.

**Non-LLM consolidation approach**:

1. **Periodic clustering** (e.g., on `decay_and_reinforce()` call):
   - Run HDBSCAN on memory embeddings (handles variable cluster sizes, noise)
   - Within each cluster, identify the highest-importance memory as "canonical"
   - Link cluster members to canonical via `associations`

2. **Deduplication**:
   - Memories with cosine similarity > 0.95 to an existing memory: merge metadata, keep the one with higher importance
   - Increment access_count on the surviving memory

3. **Archival**:
   - Memories with importance decayed below `decay_floor` AND access_count < 2 AND age > 30 days: archive (soft delete or move to cold storage)

4. **Tier promotion** (FadeMem-style):
   - High-importance memories (>0.7) get stability boost (longer half-life)
   - Low-importance memories (<0.3) get stability reduction

**This wires `decay_and_reinforce()` into action** -- currently it's a stub returning None.

**Cost**: HDBSCAN on embeddings is O(n log n). No LLM calls.

**Sources**:
- https://arxiv.org/abs/2601.18642 (FadeMem)
- https://arxiv.org/abs/2506.07398 (G-Memory, NeurIPS 2025)
- https://arxiv.org/abs/2507.03724 (MemOS)

---

### Priority 6: Emotional Context in Retrieval Scoring

**Research basis**: Mnemotree already stores `emotional_valence` and `emotional_arousal` on memories but they're unused in scoring. Cognitive science research shows emotional memories are recalled more readily (emotional enhancement of memory). The NRC VAD Lexicon v2 provides valence/arousal/dominance scores for 55K+ words without any LLM.

**Lightweight emotional scoring boost**:
```python
# In _calculate_importance_score:
emotional_boost = 0.0
if memory.emotional_arousal and memory.emotional_arousal > 0.6:
    emotional_boost = 0.05  # High-arousal memories are more salient
if memory.emotional_valence and abs(memory.emotional_valence) > 0.7:
    emotional_boost += 0.03  # Strong valence (pos or neg) enhances recall

importance += emotional_boost
```

**For ingestion (without LLM)**: Use the NRC VAD Lexicon to automatically score emotional dimensions at remember() time when no LLM analysis is available. This is a simple dictionary lookup per token, averaged across the content.

**Cost**: Dictionary lookup. No LLM calls.

**Sources**:
- https://saifmohammad.com/WebPages/nrc-vad.html (NRC VAD Lexicon v2)
- https://github.com/bagustris/text-vad

---

### Priority 7: Selective PRF (Pseudo-Relevance Feedback)

**Research basis**: PRF can hurt queries where the initial BM25 results are poor (query drift problem). Research shows that selectively applying PRF based on query performance prediction improves overall quality.

**Current state**: PRF is always-on when enabled. The `BaseQueryExpander` always attempts expansion.

**Simple selective PRF heuristic (no LLM)**:
```python
# Only apply PRF when initial BM25 results are confident
top_score = ranked[0][1] if ranked else 0.0
score_spread = top_score - (ranked[4][1] if len(ranked) > 4 else 0.0)

# Skip PRF if top results are weak (likely poor initial retrieval)
if top_score < 2.0 or score_spread < 1.0:
    return ranked  # PRF would cause drift
```

**Cost**: Zero additional computation. A conditional check on existing scores.

**Sources**:
- https://link.springer.com/article/10.1007/s10791-021-09393-5
- https://arxiv.org/html/2401.11198v1

---

### Priority 8: Contradiction Detection Between Memories

**Research basis**: DeBERTa-based NLI cross-encoders can detect contradictions between text pairs with >90% accuracy at inference speeds of ~150ms per pair. The `cross-encoder/nli-deberta-v3-xsmall` model is the lightest option.

**Approach**: At remember() time, check the top-3 most similar existing memories for contradiction:
```python
# Only for high-similarity matches (> 0.7 cosine)
for similar_memory in top_similar:
    scores = nli_model.predict([(content, similar_memory.content)])
    if scores['contradiction'] > 0.8:
        memory.conflicts_with.append(similar_memory.memory_id)
```

This populates the existing `conflicts_with` field on MemoryItem which is currently never used programmatically.

**Cost**: Small NLI model inference (~50ms per pair). No LLM calls. Optional -- only runs on high-similarity matches.

**Sources**:
- https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall

---

### Priority 9: Binary Embedding Quantization

**Research basis**: Binary quantization compresses 32-bit float embeddings to 1-bit per dimension, giving 32x storage reduction and 25x retrieval speedup while retaining >95% accuracy (with rescoring).

**Application**: For large memory stores, quantize embeddings at ingestion and use binary similarity for initial candidate recall, then rescore top candidates with full-precision embeddings.

**Cost**: CPU-only. Reduces memory footprint dramatically for in-memory indices.

**Sources**:
- https://huggingface.co/blog/embedding-quantization

---

## What NOT to Add (Avoids High-Cost LLM Dependencies)

| Approach | Why Skip |
|---|---|
| LLM-based importance scoring (Park et al.) | Requires LLM call per memory ingestion |
| LLM-based memory summarization | LLM call per consolidation cycle |
| GraphRAG with LLM extraction | LLM for entity/relationship extraction |
| Sleep-time compute (Letta) | LLM inference during idle time |
| ACAN cross-attention networks | Requires training a dedicated neural network |
| LLM-based query expansion | LLM call per retrieval query |

---

## Implementation Priority Matrix

| Priority | Change | Effort | Impact | LLM Cost |
|---|---|---|---|---|
| P1 | Unify decay to FSRS power-law | Medium | High | Zero |
| P2 | Memory-type-specific decay profiles | Low | Medium | Zero |
| P3 | Embedding-based novelty detection | Low | Medium | Zero |
| P4 | Upgrade reranker to MiniLM-L-6-v2 | Low | Medium | Zero |
| P5 | Memory consolidation pipeline | High | High | Zero |
| P6 | Emotional scoring boost | Low | Low | Zero |
| P7 | Selective PRF | Low | Low | Zero |
| P8 | Contradiction detection (NLI) | Medium | Medium | Zero (small model) |
| P9 | Binary embedding quantization | Medium | Medium (at scale) | Zero |

---

## Key Papers and Projects Referenced

| Name | Year | Key Contribution | URL |
|---|---|---|---|
| FSRS (v4-v6) | 2023-2026 | Power-law forgetting curve, DSR model | https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm |
| FadeMem | Jan 2026 | Stretched exponential, dual-layer memory | https://arxiv.org/abs/2601.18642 |
| G-Memory | Jun 2025 | Three-tier hierarchical graph memory (NeurIPS 2025) | https://arxiv.org/abs/2506.07398 |
| MemOS | Jul 2025 | Memory operating system, MemCubes | https://arxiv.org/abs/2507.03724 |
| MEM1 | Jun 2025 | RL-trained constant-memory consolidation | https://arxiv.org/abs/2506.15841 |
| ACAN | 2025 | Cross-attention for memory retrieval | https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full |
| SimpleMem | Jan 2026 | Semantic lossless compression | https://arxiv.org/abs/2601.02553 |
| TiMem | Jan 2026 | Temporal memory tree consolidation | https://arxiv.org/html/2601.02845v1 |
| Zep/Graphiti | 2025 | Temporal knowledge graph for agent memory | https://arxiv.org/abs/2501.13956 |
| AFM | Nov 2025 | Adaptive Focus Memory, fidelity levels | https://arxiv.org/html/2511.12712 |
| Memory in the Age of AI Agents | Dec 2025 | Comprehensive survey | https://arxiv.org/abs/2512.13564 |
| Drift-Adapter | 2025 | Near zero-downtime embedding migration | https://aclanthology.org/2025.emnlp-main.805 |
| Kahana & Adler | 2002 | Power-law aggregate forgetting | https://memory.psych.upenn.edu/files/pubs/KahaAdle02.pdf |
| NRC VAD Lexicon v2 | 2025 | 55K word valence/arousal/dominance scores | https://saifmohammad.com/WebPages/nrc-vad.html |
| Park et al. | 2023 | Generative Agents memory architecture | https://arxiv.org/abs/2304.03442 |
