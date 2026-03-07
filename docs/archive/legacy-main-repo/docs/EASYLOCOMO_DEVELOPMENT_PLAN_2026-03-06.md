# EasyLocomo Development Plan

Date: March 6, 2026

## Goal

Raise Mnemotree's EasyLocomo benchmark performance by improving retrieval quality, fact representation, and temporal reasoning before expanding broader "memory platform" features.

## Current Benchmark Position

- Best full-run result in the current repo: about 52.3% strict accuracy and 62.3% partial-credit accuracy.
- The strongest configuration already uses:
  - `BAAI/bge-base-en-v1.5`
  - `cross-encoder/ms-marco-MiniLM-L-12-v2`
  - hybrid retrieval
  - broad candidate pools
  - dedup disabled
  - intent filter disabled
- Current failure pattern is mostly retrieval incompleteness and temporal ambiguity, not total answer-generation failure.

## Primary Thesis

The next major gains will not come from generic consolidation, decay, or memory-governance features.

The next major gains should come from:

1. Turning extracted facts into first-class searchable objects.
2. Making temporal state a core part of ingestion and retrieval.
3. Keeping recall broad and avoiding destructive pre-filtering.
4. Using graph expansion selectively for multi-hop and relational questions.
5. Improving retrieval granularity and context assembly for list and attribution questions.

## Working Assumptions

- Benchmark-facing implementation should land in `mnemotree-develop` first because the EasyLocomo harness imports that tree.
- Full benchmark score is the main merge criterion.
- Category 5 should not be over-optimized because its current evaluator is a shallow "not mentioned" match.

## Success Metrics

### Benchmark Targets

- Milestone A: 55%+ strict accuracy on the full 1,986-question run.
- Milestone B: 58%+ strict accuracy on the full run.
- Stretch: 60%+ strict accuracy without large regression on category 4.

### Category Targets

- Category 1 (single-hop): move from about 30% toward 38%+.
- Category 2 (temporal): move from about 38% toward 45%+.
- Category 3 (multi-hop): move from about 22% toward 30%+.
- Category 4 (open-domain): do not regress by more than 1.5 percentage points.

### Diagnostic Targets

- Reduce "partial because list is incomplete" failures.
- Reduce "wrong nearby fact" retrieval failures.
- Reduce recap-date substitution errors such as returning the recap session date instead of the original event date.

## Benchmark Control Phase

### Objective

Freeze one canonical benchmark configuration so future gains are attributable to code changes rather than benchmark drift.

### Tasks

- Create a canonical EasyLocomo preset based on the strongest current stack:
  - `BAAI/bge-base-en-v1.5`
  - `cross-encoder/ms-marco-MiniLM-L-12-v2`
  - hybrid retrieval
  - category-aware `k`
  - `rerank_candidates=100`
  - dedup disabled
  - intent filter disabled
  - PRF enabled
- Keep two benchmark loops:
  - fast slice: 199-question `conv-26` run
  - full run: 1,986 questions
- Extend benchmark output with retrieval diagnostics:
  - whether gold evidence appeared in top-20
  - whether gold evidence appeared in top-100
  - retrieval source of each candidate: vector, BM25, entity, graph, fact
  - whether any filtering stage removed answer-bearing candidates

### Expected Files

- `mnemotree-test/EasyLocomo/mnemotree_locomo_optimized.py`
- `mnemotree-test/EasyLocomo/evaluate_with_judge.py`
- new benchmark analysis utility under `mnemotree-test/EasyLocomo/`

### Exit Criteria

- Reruns are stable within about +/- 0.5 percentage points.
- Every benchmark result captures enough provenance to explain failures.

## Phase 1: First-Class Fact Memories

### Objective

Promote extraction output from side metadata into searchable atomic fact objects.

### Why

The current extractor is expensive, but its output is only weakly integrated into recall. This leaves Mnemotree vulnerable to "nearby but wrong" retrieval.

### Tasks

- Add a structured fact memory layer in `MemoryCore` ingestion.
- Emit one or more atomic facts from each eligible turn.
- Preserve raw turns. Do not replace them.
- Each fact memory should include:
  - `source_memory_id`
  - `fact_type`
  - `subject`
  - `predicate`
  - `object` or `value`
  - `speaker`
  - `confidence`
  - `event_time`
  - `reference_time`
  - `session`
  - `memory_layer="fact"`
- Add storage and retrieval support for mixed memory layers:
  - raw turns
  - fact memories
  - optional windows

### Implementation Notes

- Start from the existing extracted fact hook in `mnemotree-develop/src/mnemotree/core/memory.py`.
- Replace the current "single free-text fact" behavior with multi-fact structured persistence.
- Keep provenance explicit so answer assembly can cite supporting raw turns.

### Validation

- Build a small error set of single-hop and temporal failures.
- Measure whether the gold fact appears in top-20 more often after this phase.
- Target: +2 overall points or +4 combined points across categories 1 and 2.

## Phase 2: Temporal Truth as Core Infrastructure

### Objective

Make time normalization and temporal validity part of the stored memory model, not just pre-processing text.

### Why

Many current misses are not pure retrieval failures. They are failures to distinguish:

- event time vs mention time
- original event vs recap mention
- current truth vs historical truth
- exact date vs approximate temporal relation

### Tasks

- Move temporal normalization into the main ingestion path.
- Add durable temporal fields:
  - `event_time`
  - `mention_time`
  - `reference_date`
  - `valid_from`
  - `valid_to`
  - `time_granularity`
  - `temporal_confidence`
- Add retrieval modes or scoring boosts for:
  - `when`
  - `before`
  - `after`
  - `still`
  - `currently`
  - `planning`
- Prefer original event mentions over recap mentions for date questions.
- Preserve both normalized form and original surface form for answer generation.

### Expected Files

- `mnemotree-develop/src/mnemotree/core/memory.py`
- `mnemotree-develop/src/mnemotree/normalization/`
- `mnemotree-develop/src/mnemotree/core/retrieval.py`
- `mnemotree-develop/src/mnemotree/core/scoring.py`

### Validation

- Add a temporal failure suite from real EasyLocomo misses.
- Target: category 2 reaches 45%+ on the full run.

## Phase 3: Retrieval Cleanup Without Early Narrowing

### Objective

Improve ranking while keeping candidate recall broad.

### Why

EasyLocomo currently rewards systems that surface all relevant evidence first and prune later. Aggressive early filtering loses answer-bearing candidates.

### Tasks

- Change intent classification from hard filter to soft ranking feature.
- Change dedup from destructive merge into lineage-aware aliasing:
  - preserve original turns
  - mark near-duplicates
  - optionally collapse only at presentation time
- Make PRF selective instead of unconditional.
- Tighten entity retrieval for common names:
  - cap entity candidate counts
  - require stronger semantic agreement
  - downweight broad entity-only matches
- Return retrieval explanations for debugging:
  - score components
  - source contribution
  - filters applied

### Expected Files

- `mnemotree-develop/src/mnemotree/core/memory.py`
- `mnemotree-develop/src/mnemotree/core/retrieval.py`
- `mnemotree-develop/src/mnemotree/core/_internal/deduplication.py`
- `mnemotree-develop/src/mnemotree/core/_internal/indexing.py`
- `mnemotree-develop/src/mnemotree/core/intent.py`

### Validation

- Compare against the canonical benchmark preset.
- Target: +1.5 to +3 overall points with no meaningful category 4 regression.

## Phase 4: Selective Graph Retrieval

### Objective

Make graph retrieval a real ranking stage, but only when the query actually benefits from graph expansion.

### Why

Current graph evidence in the repo shows clear value for multi-hop workloads, but slight harm on non-multi-hop queries.

### Tasks

- Generate actual `graph_candidates` in the hybrid retrieval path.
- Use graph expansion only for:
  - multi-hop questions
  - relational questions
  - selected temporal queries
- Seed graph retrieval from strong vector, BM25, or fact hits.
- Expand through:
  - `PART_OF`
  - entity co-occurrence
  - temporal adjacency
  - contradiction / supersession links
  - parent-child fact provenance
- Fuse graph signals with vector, BM25, and reranker scores.

### Expected Files

- `mnemotree/src/mnemotree/core/hybrid_retrieval.py`
- `mnemotree-develop/src/mnemotree/core/retrieval.py`
- store implementations that support graph traversal

### Validation

- Run multi-hop-only slice and full benchmark.
- Target: category 3 gains +5 points without broad regression elsewhere.

## Phase 5: Retrieval Granularity and Context Assembly

### Objective

Represent conversation evidence at the right unit for question answering.

### Why

Single turns are often too narrow, while session summaries are too lossy.

### Tasks

- Add overlapping 2-3 turn windows in addition to raw turns and facts.
- Retrieve a mix of:
  - atomic facts for precision
  - windows for conversational context
  - raw turns for provenance
- For list questions, enforce session diversity in the final candidate pool.
- For attribution questions, boost windows that preserve speaker contrast.
- Build final answer context in this order:
  - fact memories
  - supporting raw/window memories
  - temporal anchors

### Expected Files

- `mnemotree-develop/src/mnemotree/core/memory.py`
- `mnemotree-develop/src/mnemotree/core/retrieval.py`
- benchmark-side prompt assembly in `mnemotree-test/EasyLocomo/mnemotree_locomo_optimized.py`

### Validation

- Inspect current category 1 partials involving incomplete lists.
- Target: category 1 reaches 38%+ strict accuracy.

## Phase 6: Broader Memory Platform Work

### Objective

Only after benchmark-critical retrieval work, resume broader roadmap work.

### Includes

- explicit write actions
- pinned core memory tier
- durable truth maintenance
- scheduled consolidation
- spaced repetition persistence

### Rule

Do not let these items displace Phases 1-5 until the benchmark reaches at least Milestone B.

## Suggested Delivery Order

1. Benchmark control and diagnostics.
2. First-class fact memories.
3. Temporal truth infrastructure.
4. Retrieval cleanup without early narrowing.
5. Selective graph retrieval.
6. Retrieval granularity and context assembly.
7. Broader memory-platform work.

## Risks

### Risk 1: Over-abstraction

Session summaries and entity profiles can hide exact evidence and hurt precision.

Mitigation:

- keep raw turns
- keep fact provenance
- never rely on summaries alone

### Risk 2: Destructive deduplication

Merging near-duplicate turns can erase benchmark-relevant distinctions.

Mitigation:

- preserve originals
- store duplicate relationships instead of overwriting evidence

### Risk 3: Overfitting to category 5

Current category 5 evaluation is easier than other categories.

Mitigation:

- optimize primarily for categories 1, 2, and 3
- treat category 5 as a sanity check, not a primary target

### Risk 4: Benchmark drift

Frequent config changes can make score deltas meaningless.

Mitigation:

- lock one canonical benchmark preset
- record all config fields in result output

## Deliverables

### Deliverable A

Benchmark harness with retrieval diagnostics and fixed canonical preset.

### Deliverable B

Structured fact-memory ingestion with provenance-preserving storage.

### Deliverable C

Temporal-aware retrieval and scoring.

### Deliverable D

Soft retrieval heuristics replacing hard pre-filters.

### Deliverable E

Conditional graph retrieval for multi-hop and relational questions.

### Deliverable F

Mixed-granularity context assembly for final QA.

## Near-Term Sprint Breakdown

### Sprint 1

- freeze benchmark preset
- add provenance logging
- add failure taxonomy script
- define fact-memory schema

### Sprint 2

- implement multi-fact extraction and storage
- add mixed-layer retrieval
- benchmark on 199-question slice

### Sprint 3

- add temporal fields and temporal-aware retrieval
- benchmark on full run

### Sprint 4

- remove hard intent filtering
- make dedup non-destructive
- add selective PRF

### Sprint 5

- integrate conditional graph retrieval
- add window-based context assembly
- run full comparison and write results note

## Final Guidance

For EasyLocomo, Mnemotree should behave less like a generic memory vault and more like a fact-grounded temporal retrieval engine with provenance.

That means the immediate development center should be:

- facts
- time
- broad retrieval
- selective graph use
- mixed-granularity context

It should not yet be:

- consolidation-first
- decay-first
- governance-first
- summary-first
