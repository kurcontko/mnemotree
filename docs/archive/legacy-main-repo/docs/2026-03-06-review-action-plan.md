# March 6 Review and Action Plan

Date: March 6, 2026

## Reviewed Documents

- `docs/6-march.md`
- `docs/2026-03-06-benchmark-experiment-priorities.md`
- `docs/EASYLOCOMO_DEVELOPMENT_PLAN_2026-03-06.md`

## Review Summary

The three March 6 documents are mostly aligned, but they operate at different levels:

- `6-march.md` is the long-horizon product roadmap for Mnemotree as a memory operating layer.
- `2026-03-06-benchmark-experiment-priorities.md` is the near-term benchmark experiment order.
- `EASYLOCOMO_DEVELOPMENT_PLAN_2026-03-06.md` is the implementation-heavy benchmark roadmap.

The core conclusions are sound:

- benchmark progress should come from retrieval quality, temporal grounding, and better context assembly before broad platform expansion
- the write path is still simpler than the long-term product vision
- normalization and temporal handling exist, but are not yet a first-class default path in the core library
- graph retrieval should not be the first lever for the current EasyLocomo bottlenecks

## Key Gaps Confirmed In `mnemotree`

As of this review, the main repo still has these benchmark-relevant gaps:

- `src/mnemotree/core/memory.py` still stores through `_remember_sync()` without a built-in write-decision stage.
- `src/mnemotree/normalization/` exists, but the normalizer is not wired into the default `MemoryCore.remember()` ingestion path.
- `src/mnemotree/core/hybrid_retrieval.py` still calls the fusion pipeline with `graph_candidates=None`.
- Temporal fields such as `event_time`, `mention_time`, `valid_from`, and `valid_to` are not yet first-class fields in the main memory model.

There are also two practical sequencing issues:

1. The benchmark priorities note says dual-layer fact extraction should be deferred, but the EasyLocomo development plan schedules first-class fact memories as Phase 1.
2. The optimized EasyLocomo harness currently imports `mnemotree-develop/src`, not `mnemotree/src`, so benchmark gains do not automatically come from changes in this repo.

There is one positive update relative to the March 6 notes:

- retriever injection and `HybridRetriever.recall()` work appear to already be partly in flight in the current working tree, so retrieval unification should be treated as stabilization work, not a greenfield redesign

## Decisions

The action plan for `mnemotree` should resolve the document set this way:

1. Benchmark movement is the gate for broader platform work.
2. Main-repo work should focus on reusable library capabilities that help both product quality and EasyLocomo.
3. Benchmark-only prompt and model experiments should stay downstream in `mnemotree-test` unless they require a reusable library hook.
4. Structured fact memories should not be the first merge into `mnemotree`; they should be gated behind cheaper retrieval and temporal experiments.
5. Write actions, truth maintenance, consolidation, spaced repetition persistence, and broad graph work should wait until benchmark control and diagnostics are stable.

## Action Plan

### Phase 0: Freeze The Benchmark Contract

Duration: 2-3 days

Primary goal:

- define the supported library surface that the benchmark harness depends on

Tasks in `mnemotree`:

- document one canonical benchmark-facing retrieval profile based on the current strongest stack
- expose stable builder or config knobs for normalization, candidate breadth, reranking, and explanation output
- add a small benchmark integration note that states what belongs in the library versus the harness

Downstream dependency:

- `mnemotree-test/EasyLocomo/mnemotree_locomo_optimized.py` should consume the same config contract rather than re-encoding behavior ad hoc

Exit criteria:

- one documented benchmark profile exists
- downstream harness can state exactly which library settings it is using

### Phase 1: Wire Normalization Into Core Ingestion

Duration: 4-5 days

Primary goal:

- make temporal and coreference normalization a supported ingestion feature in the main library

Likely files:

- `src/mnemotree/core/builder.py`
- `src/mnemotree/core/memory.py`
- `src/mnemotree/normalization/`

Tasks:

- add an optional normalizer pipeline to `MemoryCoreBuilder` and `MemoryCore`
- allow `remember()` to preserve both raw and normalized text
- carry `reference_date` and related temporal provenance through metadata or context in a stable shape
- add tests for normalization-enabled ingestion

Exit criteria:

- the benchmark harness no longer needs to normalize content entirely outside the library
- ingestion behavior is test-covered and can be enabled explicitly

### Phase 2: Stabilize One Retrieval Path And Add Provenance

Duration: 4-5 days

Primary goal:

- finish the retrieval unification work far enough that Mnemotree has one clear hybrid path with debuggable outputs

Likely files:

- `src/mnemotree/core/memory.py`
- `src/mnemotree/core/retrieval.py`
- `src/mnemotree/core/hybrid_retrieval.py`

Tasks:

- finish and test the retriever injection path already underway
- keep one supported hybrid retrieval flow for `MemoryCore.recall()`
- expose retrieval provenance or explanation data for benchmark debugging
- convert any hard early narrowing that hurts recall into soft ranking signals where practical

Exit criteria:

- one hybrid retrieval path is clearly the supported path
- benchmark failures can be diagnosed with source and score provenance

### Phase 3: Add Low-Risk Context Assembly Hooks

Duration: 1 week

Primary goal:

- support the cheapest high-upside single-hop and temporal improvements before building new memory layers

Likely files:

- `src/mnemotree/core/memory.py`
- `src/mnemotree/core/hybrid_retrieval.py`
- `src/mnemotree/core/models.py`

Tasks:

- add adjacency or neighboring-turn expansion hooks after initial retrieval
- support optional overlapping window memories without removing raw turns
- add retrieval metadata that distinguishes raw turns, windows, and future fact memories

Exit criteria:

- downstream benchmark runs can test adjacency expansion and window retrieval without forking core logic
- raw-turn provenance remains intact

### Phase 4: Run The First Three Downstream Experiments

Duration: parallel with Phases 1-3

These should happen in `mnemotree-test` and `mnemotree-develop`, not in `mnemotree`:

1. `v11` retrieval stack plus GPT-5.2
2. `v11` plus adjacency expansion
3. `v11` plus temporal grounding block before answer generation

Rules:

- run the 199-question slice first
- only run the full benchmark after the slice shows directional gain
- do not merge new core complexity into `mnemotree` unless one of these cheaper experiments plateaus

Success gate:

- reproduce the current control before claiming improvement
- beat the control on temporal or single-hop without broad regression

### Phase 5: Reassess Whether Fact Memories Are Needed

Duration: after Phases 1-4

Primary goal:

- only start structured fact-memory work if cheaper retrieval and temporal improvements do not get the benchmark close to target

Tasks:

- review the post-experiment failure taxonomy
- check whether misses are still mostly "nearby but wrong" retrieval errors
- only then design a first-class fact-memory schema behind a feature flag

Decision gate:

- if adjacency plus temporal grounding moves the benchmark materially, defer fact memories
- if the same single-hop and temporal misses remain, start a minimal fact layer with strict provenance

### Phase 6: Resume Broader Platform Roadmap

Start only after benchmark control is stable and near-term targets are met.

Defer until then:

- explicit write actions
- pinned core versus archival tiers
- durable truth maintenance
- scheduled consolidation
- spaced repetition persistence
- graph retrieval as a broad ranking stage

## Recommended Sequence

1. Freeze the benchmark contract in `mnemotree`.
2. Wire normalization into the main ingestion path.
3. Stabilize one hybrid retrieval path with explanation output.
4. Add adjacency and mixed-granularity hooks.
5. Run the three cheaper downstream benchmark experiments.
6. Decide whether fact memories are still necessary.
7. Return to the broader memory-platform roadmap only after benchmark progress is demonstrated.

## Immediate Next Moves

If the team wants one concrete next sprint for `mnemotree`, it should be:

1. add normalization support to `MemoryCoreBuilder` and `MemoryCore`
2. add retrieval provenance output to the supported hybrid retriever
3. document the benchmark integration contract used by the EasyLocomo harness

That sprint is the most defensible bridge between the March 6 benchmark notes and the March 6 product roadmap.
