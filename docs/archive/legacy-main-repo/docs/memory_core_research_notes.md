# MemoryCore Review + Low-Cost Improvements (2026-02-10)

## Scope
This note summarizes the current MemoryCore design and proposes improvements inspired by recent memory research, with a constraint to avoid high-cost, multi-LLM pipelines.

## Current MemoryCore: Snapshot (code-based)
- Retrieval: vector similarity with optional entity recall; optional BM25 + PRF; optional RRF fusion and FlashRank reranking.
  - See: `src/mnemotree/core/retrieval.py`, `src/mnemotree/core/_internal/indexing.py`
- Enrichment: always embeddings; optional NER/keywords; optional LLM analysis + summarization in pro mode.
  - See: `src/mnemotree/core/_internal/enrichment.py`, `src/mnemotree/analysis/memory_analyzer.py`
- Scoring: importance + recency + relevance; optional decay; access-based reinforcement.
  - See: `src/mnemotree/core/scoring.py`, `src/mnemotree/core/models.py`
- Gaps:
  - `decay_and_reinforce()` is a stub.
  - Scoring currently *filters* by threshold more than it *ranks* results.
  - Filters are applied post-retrieval even when stores could support structured queries.
  - No global memory consolidation / replay loop.
  - No lightweight graph-based retrieval signal beyond entities.

## Research-Inspired Improvements (low cost)

### 1) Implement a real decay + spaced-retrieval loop (no LLM required)
**Research basis:** MemoryBank uses a forgetting-curve inspired update mechanism. Spaced retrieval practice improves long-term retention in humans.
- Sources: https://arxiv.org/abs/2305.10250, https://www.nature.com/articles/s44159-022-00089-1

**Implementation sketch:**
- Implement `MemoryCore.decay_and_reinforce()` to periodically:
  - Compute decayed importance using existing `MemoryScoring` parameters.
  - Select a small batch of “at-risk” memories (low access_count, high age, low similarity hit rate).
  - Apply light reinforcement by bumping importance or access_count.
- This is a background task using current metadata—no LLM calls.

### 2) Prioritized replay for weak / novel memories
**Research basis:** Hippocampal replay prioritizes weak or novel items; replay predicts later memory strength.
- Source: https://www.nature.com/articles/s41467-023-43939-z

**Implementation sketch:**
- Maintain a “priority score” = novelty + recency + low-access weight.
- Periodically sample top-k for replay and apply a small reinforcement bump.
- Optional: re-embed only if content changes (avoid extra cost).

### 3) Lightweight entity-graph recall (HippoRAG-style without heavy LLMs)
**Research basis:** HippoRAG uses a KG + Personalized PageRank to improve multi-hop retrieval.
- Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html

**Implementation sketch:**
- Build an entity co-occurrence graph from existing `entities` / `linked_concepts` fields.
- Run Personalized PageRank seeded by query entities and fuse candidates via RRF.
- This is mostly graph math + existing data (no LLM).

### 4) Short-term “working memory” cache (kNN-style)
**Research basis:** Memorizing Transformers use an external memory for recent representations.
- Source: https://iclr.cc/virtual/2022/poster/6064

**Implementation sketch:**
- Keep a small in-memory ANN index for last N memories per user/conversation.
- Fuse top-k from this cache into recall, then de-duplicate with existing vector results.
- No extra models; uses existing embeddings.

### 5) Global memory summaries with minimal cost
**Research basis:** MemoRAG shows a global memory + dual-system pipeline, but we can implement a light version.
- Source: https://arxiv.org/abs/2409.05591

**Implementation sketch:**
- Maintain cluster centroids (already have clustering utilities).
- Store short extractive summaries or median-of-embeddings “gist” strings.
- Refresh only when cluster drift crosses a threshold.

## Low-Cost Engineering Tweaks (immediate wins)
1) **Rank by score, not only filter:** in `MemoryScoring.filter_memories_by_score`, sort by score and then filter/threshold.
2) **Push filters down when possible:** if store supports structured queries, apply filters before retrieval instead of post-filtering.
3) **Expose retrieval configuration defaults explicitly:** make `recency_stability_seconds`, `recency_power`, and decay knobs configurable in builder.

## Recommended Next Steps (minimal surface area)
- Implement `decay_and_reinforce()` + a background scheduler hook.
- Add a lightweight replay policy (sampling based on novelty + access stats).
- Add entity-graph recall as an optional retrieval stage fused via RRF.
- Adjust scoring to rank + threshold.

## Notes on Cost Control
- All improvements above can run without additional LLM calls.
- LLM usage is optional and should remain confined to pro-mode enrichment/summarization.
