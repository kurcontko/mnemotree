# Mnemotree Benchmark Experiment Priorities

Date: March 6, 2026

## Scope

This note focuses on which experiments to run first to close benchmark gaps, with emphasis on:

- single-hop retrieval
- temporal reasoning

This is a planning document only. No implementation work is proposed here.

## Executive Summary

The current evidence says single-hop and temporal should not be treated as one problem.

- Single-hop is mainly a retrieval completeness and precision problem.
- Temporal is mainly a temporal grounding and answer-generation problem.

The best immediate path is:

1. Keep the current strongest retrieval stack.
2. Re-run it with a stronger answer model for temporal recovery.
3. Run targeted retrieval experiments for single-hop completeness.
4. Only then test heavier ingestion changes such as fact extraction or ENGRAM-style memory.

## Current Evidence

### Old baseline is no longer the right reference point

The original GPT-4.1 run was weak largely because important retrieval features were disabled:

- strict accuracy: 33.7%
- single-hop: 17.4%
- temporal: 43.9%

That run had hybrid retrieval, NER, and keywords disabled.

Reference:

- `/home/qrc/repos/BENCHMARK_RESULTS/mnemotree_gpt41_benchmark_report.md`

### The stronger retrieval stack is now in the EasyLocomo runs

The newer benchmark harness already applies:

- hybrid retrieval
- BM25 + RRF fusion
- reranking
- normalization
- category-aware prompting
- per-sample store isolation

The strongest full runs so far are:

| Run | Accuracy | Single-hop | Temporal | Notes |
|-----|----------|------------|----------|-------|
| v9_develop | 43.6% | 18.8% | 24.6% | develop regression case |
| v10_develop | 51.4% | 27.0% | 37.7% | major recovery from better retrieval stack |
| v11_k20 | 52.3% | 29.8% | 38.0% | best current full run |

References:

- `/home/qrc/repos/mnemotree-test/EasyLocomo/outputs/v9_develop/evaluated.json`
- `/home/qrc/repos/mnemotree-test/EasyLocomo/outputs/v10_develop/evaluated.json`
- `/home/qrc/repos/mnemotree-test/EasyLocomo/outputs/v11_k20/evaluated.json`

### What the deltas say

From v9 to v10:

- overall: +7.9pp
- single-hop: +8.2pp
- temporal: +13.1pp

This is the strongest evidence that infrastructure and retrieval defaults matter first.

From v10 to v11:

- overall: +0.9pp
- single-hop: +2.8pp
- temporal: +0.3pp

This matters a lot:

- raising effective retrieval breadth still helps single-hop
- it does almost nothing for temporal

So temporal is no longer recall-limited in the same way single-hop is.

### Retrieval-only evals support the same conclusion

On the local retrieval benchmark, `rrf + bm25` lifts MRR from 0.8063 to 0.9426.

References:

- `/home/qrc/repos/mnemotree/benchmarks/results/evaluation_baseline.json`
- `/home/qrc/repos/mnemotree/benchmarks/results/eval_rrf_bm25_full.json`

This strongly supports continuing to invest in exact-match and fusion retrieval for single-hop.

## Diagnosis By Category

### Single-hop

Observed pattern:

- single-hop remains the weakest practical retrieval category even in the best current full run
- the partial count is still very high
- increasing retrieval breadth helped

In `v11_k20`:

- correct: 84
- partial: 128
- incorrect: 70

Interpretation:

- the system often retrieves the right region of context
- the answer is frequently incomplete, overly broad, or not focused on the asked fact
- this points to chunk shape, neighboring context recovery, and ranking precision

Likely causes:

- facts are spread across adjacent turns, but turns are stored individually
- semantic retrieval is still competing with distractor memories
- entity and BM25 signals help, but not enough to fully assemble list-style answers

### Temporal

Observed pattern:

- temporal improved a lot when the retrieval stack recovered from the v9 regression
- temporal barely improved when k was pushed harder
- earlier GPT-5.2 runs with weaker retrieval still outperformed the current DeepSeek-based temporal score

Relevant scores:

- v4_gpt52 temporal: 46.4%
- v5_ner_keywords temporal: 48.3%
- v6_full temporal: 46.4%
- v11_k20 temporal: 38.0%

Interpretation:

- temporal is not primarily blocked by missing recall anymore
- the main issue is converting retrieved evidence into the right date, interval, or relative time answer
- the answer model is likely the current bottleneck once retrieval is "good enough"

Likely causes:

- the model confuses event date with conversation date
- the model tends to answer with approximate dates when the benchmark expects a relative expression or a specific year
- temporal normalization helps retrieval, but not final temporal selection

## Structural Notes In Main Mnemotree

These product-level gaps matter because the benchmark harness already compensates for them externally.

### Normalization exists but is not wired into core ingestion

The normalization pipeline is implemented, including temporal normalization, but the main `MemoryCore.remember()` path does not apply it by default.

References:

- `/home/qrc/repos/mnemotree/src/mnemotree/normalization/__init__.py`
- `/home/qrc/repos/mnemotree/src/mnemotree/core/memory.py`
- `/home/qrc/repos/mnemotree-test/EasyLocomo/mnemotree_locomo_optimized.py`

### Graph retrieval is still not active in the hybrid path

The hybrid retriever still calls `retrieve(..., graph_candidates=None, ...)`.

Reference:

- `/home/qrc/repos/mnemotree/src/mnemotree/core/hybrid_retrieval.py`

This means graph work should not be the first experiment for the current single-hop and temporal gaps.

## Recommended Experiment Order

### P0. Freeze the current strong retrieval baseline

Use the `v11` shape as the control:

- bge-base embeddings on CUDA
- L-12 cross-encoder reranker
- hybrid retrieval on
- NER on
- keywords on
- normalization on
- dedup off
- intent filter off
- per-sample isolation

Reason:

- this is the strongest full retrieval configuration already demonstrated
- it avoids conflating new experiments with known regressions

Success criterion:

- reproduce ~52% overall, ~30% single-hop, ~38% temporal before changing anything else

### P1. Re-run the `v11` stack with GPT-5.2

This should be the first experiment.

Reason:

- temporal is the main target where retrieval gains have plateaued
- prior GPT-5.2 runs were materially better on temporal even with weaker retrieval
- this isolates model-side temporal reasoning from retrieval-side effects

Expected effect:

- strongest gain on temporal
- likely modest gain on multi-hop
- little direct effect on single-hop retrieval quality, though some partials may convert to correct

Success criterion:

- temporal should move back toward the mid-to-high 40s
- overall should exceed the current `v11` control by a meaningful margin

### P2. Run adjacency expansion for single-hop

Experiment:

- when a memory is retrieved, include one or two neighboring turns from the same session

Reason:

- single-hop partial answers strongly suggest the system finds the right area but misses the exact supporting turn
- this is cheaper and lower-risk than redesigning ingestion first

Expected effect:

- strongest gain on single-hop
- possible small gain on open-domain
- minimal gain on temporal unless the adjacent turn contains the needed date anchor

Success criterion:

- single-hop correct rate rises without major increase in off-target answers
- partial count drops more than incorrect count rises

### P3. Run multi-turn chunking for single-hop

Experiment:

- ingest overlapping 2-3 turn windows instead of only atomic turns

Reason:

- many single-hop questions ask for a specific fact that is stated across a short exchange, not one isolated utterance
- this directly addresses list completeness and local context fragmentation

Why this is after adjacency expansion:

- adjacency expansion is lower effort and easier to reason about
- if adjacency already captures most of the gain, chunking may be unnecessary

Success criterion:

- improves single-hop over the P2 result, not just over the old baseline

### P4. Add temporal grounding before answer generation

Experiment:

- extract normalized time candidates from retrieved memories
- build a small structured timeline block
- ask the answer model to choose from that grounded timeline rather than infer directly from raw turns

Reason:

- temporal errors are frequently wrong-year, wrong-anchor, or conversation-date substitution errors
- increasing `k` did not materially improve temporal, so the remaining gap is downstream of retrieval

Expected effect:

- strongest gain on temporal
- limited effect on other categories

Success criterion:

- temporal correct improves without a large increase in partials
- fewer errors where the predicted date matches the conversation date instead of the event date

### P5. Run small retrieval ablations only after the above

Experiments:

- PRF off
- restore earlier fusion weights
- reduce entity boost

Reason:

- these matter, but they are secondary once the large regressions are already removed
- they should be measured from the strong control, not from weak runs

Success criterion:

- keep any ablation only if it improves the target category without degrading the others materially

## Experiments To Defer

### Dual-layer fact extraction

Do not prioritize this first.

Reason:

- `v8_factex_ds32` regressed both single-hop and temporal relative to `v7` and `v10`
- it adds complexity and another model dependency before the current bottlenecks are isolated

### ENGRAM-style fact memory experiments

Do not prioritize this first.

Reason:

- existing ENGRAM runs are small-sample only
- they do not yet show a clear single-hop win
- temporal may improve later from this direction, but the evidence is not strong enough yet

### Graph retrieval

Do not prioritize this first.

Reason:

- graph is not even fully active in the main hybrid path yet
- it is more likely to help multi-hop and relational questions than the current single-hop and temporal bottlenecks

## Concrete Recommended Sequence

If only three experiments are run next, they should be:

1. `v11` retrieval stack + GPT-5.2
2. `v11` + adjacency expansion
3. `v11` + temporal grounding block

If five experiments are run next, use:

1. `v11` retrieval stack + GPT-5.2
2. `v11` + adjacency expansion
3. `v11` + multi-turn chunking
4. `v11` + temporal grounding block
5. `v11` + PRF / fusion ablations

## What Success Looks Like

### Single-hop

Target near-term movement:

- from ~30% toward mid-30s or better

The key indicator is not just higher accuracy. It is:

- fewer partials
- more exact answers
- fewer broad list answers with irrelevant extras

### Temporal

Target near-term movement:

- from ~38% back toward the 46-48% range already seen in earlier GPT-5.2 runs

The key indicator is:

- fewer conversation-date substitutions
- fewer approximate answers when a specific grounded answer exists

## Final Recommendation

The first thing to optimize is not a new memory architecture.

The first thing to optimize is the benchmark path that is already closest to working:

- keep the strong retrieval stack
- improve the answer model for temporal
- improve local context assembly for single-hop

Only after those are measured cleanly should the team spend time on:

- fact extraction
- ENGRAM-style memory
- graph-aware retrieval

That order is the most defensible path to closing the single-hop and temporal gaps without mixing too many variables at once.
