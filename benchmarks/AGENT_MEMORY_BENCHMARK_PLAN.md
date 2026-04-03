# Agent Memory Benchmark Expansion Plan

## Goal

Extend Mnemotree's benchmark coverage beyond the current internal dataset and LoCoMo-style runs so we can measure:

- long-term conversational recall
- temporal and update handling
- personalized preference memory
- hallucination in memory extraction and updates
- agentic memory use across multi-step tasks

The plan below is ordered by implementation value, fit with the current codebase, and integration risk.

## Current Baseline

Mnemotree already has two evaluation shapes:

1. `benchmarks/evaluate.py`
   - Best for retrieval and optional answer-level QA over normalized `memories.jsonl` and `test_queries.jsonl`.
2. Dedicated LoCoMo scripts under `mnemotree-test/EasyLocomo/`
   - Best for session-based conversational benchmarks that need custom ingestion and scoring.

That means the next step is not to overload `evaluate.py` with every benchmark format. The right direction is a small adapter layer plus benchmark-specific runners where needed.

## Recommended Benchmark Portfolio

### P0: Add first

#### 1. LongMemEval

Why:

- Closest fit to Mnemotree's current conversational memory shape.
- Covers information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention.
- Has official evaluation scripts and retrieval metrics.
- Updated cleaned dataset was released in September 2025, so it is current enough to use as a primary benchmark.

How to use it:

- Start with `longmemeval_oracle.json` first to validate answering and session recall without full haystack difficulty.
- Then add `longmemeval_s_cleaned.json`.
- Defer `longmemeval_m_cleaned.json` until pipeline stability is proven.

Why first:

- It can reuse most of the existing LoCoMo-style ingestion and QA flow with less adaptation than any other external benchmark.

#### 2. PersonaMem

Why:

- Strong benchmark for dynamic user profiling and personalized responses.
- Tests whether Mnemotree can track evolving user preferences, not just retrieve facts.
- Covers 180+ simulated user histories, up to 60 sessions, across 15 scenarios and 7 query types.

How to use it:

- Start with the 32k and 128k variants.
- Implement the multiple-choice/discriminative path before generative evaluation.

Why second:

- It is a good test of whether Mnemotree's memory structure helps beyond factual recall.
- It complements LongMemEval without requiring full agent-environment tooling.

### P1: Add next

#### 3. PrefEval

Why:

- Focuses on explicit and implicit user preferences in long-context conversations.
- Good for measuring preference adherence with lower integration cost than full agent benchmarks.
- Useful if Mnemotree wants to position itself as a user-aware memory layer.

How to use it:

- Implement explicit-preference and implicit-preference classification tasks first.
- Add generative tasks only after classification is stable.

Why third:

- It overlaps usefully with PersonaMem, but is narrower and cheaper to run.
- It gives a clean preference-memory score that is easy to track in regressions.

#### 4. HaluMem

Why:

- Tests hallucinations in memory extraction, updating, and memory QA.
- This exposes failures that top-k recall metrics miss.
- It is the best available benchmark for write-path correctness, not just read-path correctness.

How to use it:

- Start with memory extraction and memory update tasks.
- Add full QA and hallucination breakdown after the write/update pipeline is stable.

Why fourth:

- Very valuable, but it needs new evaluator logic for operation-level scoring.

### P2: Add after the above are stable

#### 5. MemoryAgentBench

Why:

- Designed specifically for memory agents using incremental multi-turn interactions.
- Covers four core competencies: accurate retrieval, test-time learning, long-range understanding, and conflict resolution.
- Includes newer agent-memory tasks such as EventQA and FactConsolidation.

How to use it:

- Integrate as a dedicated runner with an agent wrapper interface.
- Start with accurate retrieval and conflict resolution tasks.
- Add test-time learning after the adapter is stable.

Why later:

- It is more agent-oriented than the current harness.
- It will need an execution loop that simulates incremental conversation chunks rather than static query-over-corpus evaluation.

### P3: Stretch / forward-looking

#### 6. MemoryArena

Why:

- Best candidate for testing whether memory improves actual multi-session agent behavior.
- Covers interdependent subtasks in web navigation, planning, search, and formal reasoning.

Why later:

- It evaluates memory plus action, not just memory plus QA.
- Current Mnemotree benchmark code has no agent-environment runner abstraction yet.

#### 7. LoCoMo-Plus

Why:

- New benchmark from February 2026 targeting cognitive memory under cue-trigger semantic disconnect.
- Good for measuring implicit state, goal, and value tracking beyond surface fact recall.

Why later:

- Too new to make the first integration target.
- Better as a follow-on once LoCoMo and LongMemEval are fully reproducible in-tree.

## What To Build

### 1. Refactor the benchmark harness into shared parts

Keep `benchmarks/evaluate.py`, but extract reusable pieces into a library layer:

- `benchmarks/lib/store_factory.py`
- `benchmarks/lib/ingest.py`
- `benchmarks/lib/answering.py`
- `benchmarks/lib/metrics.py`
- `benchmarks/lib/results.py`

Reason:

- External benchmarks will need the same Mnemotree store setup, memory ingestion, answer generation, latency tracking, and result serialization.
- Right now those concerns are all mixed inside `evaluate.py`.

### 2. Add a benchmark adapter interface

Create a thin normalized interface such as:

```python
class BenchmarkAdapter(Protocol):
    name: str

    def load(self, split: str) -> BenchmarkDataset: ...
    def ingest(self, memory_core, case: BenchmarkCase) -> None: ...
    def run_case(self, memory_core, case: BenchmarkCase) -> CaseResult: ...
    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]: ...
```

Core normalized concepts:

- `ConversationSession`
- `BenchmarkCase`
- `EvidenceRef`
- `ExpectedAnswer`
- `CaseResult`

This keeps benchmark-specific parsing isolated while reusing Mnemotree runtime code.

### 3. Add a dedicated external benchmark runner

Add:

- `benchmarks/run_external.py`

Responsibilities:

- select benchmark adapter
- download or validate dataset presence
- create Mnemotree config
- run benchmark-specific ingestion and querying
- save raw predictions and aggregated metrics

Do not force every benchmark through `benchmarks/evaluate.py`.

### 4. Standardize result schema across benchmarks

Keep one top-level JSON shape:

```json
{
  "benchmark": "longmemeval",
  "split": "oracle",
  "summary": {},
  "per_case_results": [],
  "config": {},
  "metadata": {}
}
```

But let `summary` contain benchmark-specific groups:

- `retrieval`
- `qa`
- `preference`
- `update_consistency`
- `hallucination`
- `agent_task_success`
- `latency_ms`

Do not flatten all benchmarks into one fake "accuracy" number.

## Benchmark-Specific Integration Notes

### LongMemEval

Adapter type:

- Session-based conversational QA benchmark

Minimal viable implementation:

1. Treat each evaluation instance as one case.
2. Ingest `haystack_sessions` in timestamp order as episodic memories.
3. Preserve session ids and dates in metadata.
4. Retrieve top-k memories or sessions for each question.
5. Generate an answer with the retrieved context.
6. Score with the official evaluation script.
7. Also compute Mnemotree-side session recall against `answer_session_ids`.

Needed additions:

- session-aware ingestion
- session-level recall metric
- abstention handling
- adapter for official output format: `question_id`, `hypothesis`

### PersonaMem

Adapter type:

- Multi-session personalization benchmark

Minimal viable implementation:

1. Load shared context and question rows.
2. Ingest the interaction history up to `end_index_in_shared_context`.
3. Run only discriminative multiple-choice evaluation first.
4. Track accuracy by query type and by distance to latest preference mention.

Needed additions:

- multiple-choice scorer
- support for dynamic profile state tracking
- long-context truncation policy that is explicit and reproducible

### PrefEval

Adapter type:

- Preference-following benchmark

Minimal viable implementation:

1. Load explicit-preference tasks first.
2. Convert preference statements and conversation turns into memories.
3. Run classification tasks before generation tasks.
4. Report accuracy by preference type, topic, and number of turns.

Needed additions:

- classification runner
- explicit vs implicit preference tagging
- reminder baseline support for comparison

### HaluMem

Adapter type:

- Operation-level memory hallucination benchmark

Minimal viable implementation:

1. Implement extraction-only evaluation first.
2. Add update evaluation second.
3. Defer full long-context QA until the first two are stable.

Needed additions:

- write-path snapshotting
- update reconciliation checker
- hallucination taxonomy in results
- support for benchmark-provided evidence links

Important:

- This should not be forced into the same metric path as retrieval-only benchmarks.

### MemoryAgentBench

Adapter type:

- Incremental agent-memory benchmark

Minimal viable implementation:

1. Create a runner that feeds chunks incrementally.
2. After each injection chunk, allow Mnemotree to store/update memories.
3. At query time, run retrieval and answering.
4. Score per competency: AR, TTL, LRU, CR.

Needed additions:

- chunked interaction runner
- competency-level result reporting
- optional agent wrapper API for methods that do more than retrieval

### MemoryArena

Adapter type:

- Agent plus environment benchmark

Minimal viable implementation:

- Do not integrate into the first benchmark wave.
- First define an `AgentWithMemory` interface:

```python
class AgentWithMemory(Protocol):
    def observe(self, event: str, metadata: dict[str, Any] | None = None) -> None: ...
    def act(self, prompt: str, tools: list[Any] | None = None) -> str: ...
```

Only after that should MemoryArena be attempted.

## Proposed File Layout

```text
benchmarks/
  evaluate.py
  run_external.py
  lib/
    answering.py
    ingest.py
    metrics.py
    results.py
    store_factory.py
  adapters/
    base.py
    longmemeval.py
    personamem.py
    prefeval.py
    halumem.py
    memoryagentbench.py
  datasets/
    README.md
    download_longmemeval.py
    download_personamem.py
    download_prefeval.py
    download_halumem.py
    download_memoryagentbench.py
  configs/
    longmemeval_oracle.json
    longmemeval_s.json
    personamem_32k.json
    personamem_128k.json
    prefeval_explicit.json
    halumem_extract.json
    memoryagentbench_ar.json
```

## Execution Plan

### Milestone 1: Harness refactor

Target:

- Extract reusable code from `benchmarks/evaluate.py`
- Add adapter interface
- Add external runner skeleton

Success criteria:

- Internal benchmark still runs unchanged
- Existing result JSON schema remains backward compatible where possible

### Milestone 2: LongMemEval

Target:

- Add `LongMemEvalAdapter`
- Support oracle split and `S` split
- Save official-format predictions

Success criteria:

- One-command run for oracle split
- Per-question outputs + official QA score + Mnemotree session recall

### Milestone 3: PersonaMem + PrefEval

Target:

- Add personalization benchmark coverage
- Prefer exact-match / multiple-choice evaluation before generative judging

Success criteria:

- Accuracy by query type
- Accuracy by context distance / session count
- Shared config pattern across both benchmarks

### Milestone 4: HaluMem

Target:

- Add extraction and update evaluation

Success criteria:

- Hallucination breakdown is visible in results
- Regressions in write-path behavior are detectable without manual inspection

### Milestone 5: MemoryAgentBench

Target:

- Add incremental chunk runner
- Report competency-level results

Success criteria:

- Can run at least AR and CR subsets end-to-end
- Can compare Mnemotree retrieval-only vs retrieval-plus-update variants

### Milestone 6: MemoryArena or LoCoMo-Plus

Target:

- Add one next-generation benchmark after earlier ones are stable

Success criteria:

- Benchmark uses a separate agent runner instead of distorting the simpler QA harness

## Engineering Rules

### Keep exact-match and judge-based metrics separate

Do not mix:

- multiple-choice accuracy
- exact match / token F1
- LLM-judged relevance

Each benchmark should report its native metric first.

### Pin benchmark versions and judge models

For reproducibility, record:

- dataset name and exact file version
- benchmark commit or release date
- answer model
- judge model
- embedding model
- store backend
- retrieval settings

### Add tiny smoke subsets for CI

For each benchmark, create a tiny local subset with:

- 3 to 10 cases
- no external judge requirement where possible
- deterministic expected outputs

This is the only realistic way to stop adapter regressions.

### Preserve benchmark-native structure

Do not normalize away important benchmark concepts:

- LongMemEval sessions
- PersonaMem evolving preferences
- HaluMem updates and hallucination labels
- MemoryAgentBench incremental chunks

Normalization should happen only at the runner boundary.

## Recommendation Summary

If we want the best effort-to-signal ratio, the order should be:

1. LongMemEval
2. PersonaMem
3. PrefEval
4. HaluMem
5. MemoryAgentBench
6. MemoryArena
7. LoCoMo-Plus

This gives Mnemotree a balanced benchmark stack:

- conversational recall and update handling
- user preference and personalization memory
- write-path hallucination resistance
- agentic memory under incremental interaction

## Source Notes

Primary sources used for this plan:

- LongMemEval official repo and README: https://github.com/xiaowu0162/LongMemEval
- MemoryAgentBench official repo and README: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- PersonaMem official project page and repo: https://zhuoqunhao.github.io/PersonaMem.github.io/ and https://github.com/bowen-upenn/PersonaMem
- PrefEval official repo: https://github.com/amazon-science/PrefEval
- HaluMem official repo: https://github.com/MemTensor/HaluMem
- MemoryArena official project page: https://memoryarena.github.io/
- LoCoMo-Plus paper abstract, published February 11, 2026: https://arxiv.org/abs/2602.10715
