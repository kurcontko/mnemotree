# Mnemotree Improvement Plan

Date: March 6, 2026

## Thesis

Mnemotree should evolve from a good memory store into a full memory operating layer for agents.

The repo already has many of the right primitives, but they are not yet part of one coherent lifecycle:

- The write path goes from enrichment to persistence with no first-class correction or gating flow in `src/mnemotree/core/memory.py`.
- Recall filters are still applied after retrieval in `src/mnemotree/core/memory.py`.
- Consolidation, truth maintenance, and write gating live in `src/mnemotree/experimental/` instead of inside the main `MemoryCore` loop.
- The graph protocol is strong, but graph retrieval is still not a real ranking stage in `src/mnemotree/core/hybrid_retrieval.py`.

The best path is not a rewrite. The best path is to promote the existing side systems into first-class behavior.

## Current Gaps

### 1. Write quality is not first-class

`MemoryCore._remember_sync()` in `src/mnemotree/core/memory.py` enriches content, builds a `MemoryItem`, and persists it. There is no built-in `ADD`, `MERGE`, `UPDATE`, `SUPERSEDE`, or `DELETE` lifecycle.

### 2. Retrieval is stronger than memory maintenance

The recall path is reasonably capable, but the maintenance path is weak:

- `decay_and_reinforce()` in `src/mnemotree/core/memory.py` mainly performs batch decay.
- `MemoryItem.update_access()` in `src/mnemotree/core/models.py` can update stability, but the live retrieval path in `src/mnemotree/core/retrieval.py` does not pass retrievability into it.
- `AdaptiveDecaySystem` in `src/mnemotree/core/adaptive.py` keeps schedules in memory, not in persistent storage.

### 3. Experimental systems are detached from production behavior

`MemorySystemConfig.build_memory_system()` in `src/mnemotree/configs.py` constructs a consolidator, claims registry, adaptive system, and write gate, but these are returned beside `MemoryCore` rather than being integrated into its write and recall paths.

### 4. Graph support is not yet a core retrieval signal

`SupportsKnowledgeGraph` in `src/mnemotree/store/protocols.py` already supports links, backlinks, traversal, and path finding. But the hybrid retrieval path still passes `graph_candidates=None`, so graph structure is not yet fully used at ranking time.

## Plan

### Phase 1. Make write decisions first-class

Promote the write path from "store this memory" to "decide what to do with this memory."

Core changes:

- Introduce explicit write actions: `ADD`, `MERGE`, `UPDATE`, `SUPERSEDE`, `DELETE`, and `NOOP`.
- Wire `ContextAwareWriteGate` from `src/mnemotree/experimental/write_gate.py` into both `remember()` and async ingestion.
- Add an audit trail for write decisions, including novelty score, confidence, merge targets, and rejection reasons.
- Expose correction APIs and MCP tools for updating or superseding memories directly.

Expected outcome:

- Fewer duplicates.
- Better handling of corrections and changing facts.
- A write pipeline that behaves more like Mem0 and less like an append-only notebook.

### Phase 2. Add an explicit memory hierarchy

Keep the current `MemoryType` model, but add a second axis for memory tier or scope.

Recommended tiers:

- `core`: pinned profile, constraints, preferences, and agent-specific standing facts.
- `working`: recent, short-lived session context.
- `archival`: searchable long-term memory.

Core changes:

- Add tier or scope fields to memory records.
- Create APIs for managing pinned core memory separately from archival memory.
- Assemble final context as: core blocks + working memory + retrieved archival memories.

Expected outcome:

- A clearer memory model for agents.
- Less overloading of vector recall for facts that should simply stay pinned.
- A cleaner separation inspired by Letta's memory blocks and archival memory.

### Phase 3. Persist truth and temporal state

Take the claim registry concept from `src/mnemotree/experimental/truth_maintenance.py` and make it durable.

Core changes:

- Persist claims, supersession links, contradiction links, validity windows, and provenance.
- Support "latest truth" queries and "historical truth" queries as separate modes.
- Add temporal normalization as a standard part of ingestion.
- Store claim lineage even when source memories are consolidated or deprecated.

Expected outcome:

- Better handling of changing user preferences and facts.
- Real time-aware memory behavior.
- A path toward Graphiti-style temporal reasoning without forcing a graph-only architecture.

### Phase 4. Turn consolidation into a real background system

The consolidator in `src/mnemotree/experimental/consolidation.py` is a good starting point, but it needs to become a persisted background pipeline.

Core changes:

- Run consolidation as a scheduled or threshold-triggered background job.
- Persist created semantic abstractions, lineage metadata, and deprecation decisions.
- Support two consolidation modes:
  - local-first clustering and deduplication with no LLM requirement
  - optional LLM summarization for higher-quality semantic abstractions
- Add replay or review queues for high-value memories instead of only decaying them.

Expected outcome:

- Engram-style dream cycle behavior.
- Letta-style sleeptime processing, but grounded in mnemotree's current architecture.
- Controlled memory growth instead of endless accumulation.

### Phase 5. Make graph retrieval real

Use the graph store as a ranking signal, not only as a storage sidecar.

Core changes:

- Actually generate `graph_candidates` inside the hybrid retriever.
- Expand candidates using backlinks, path traversal, entity neighborhoods, and claim relationships.
- Fuse vector, BM25, graph, and truth signals with provenance.
- Add recall modes such as `semantic`, `relational`, `temporal`, and `latest_truth`.

Expected outcome:

- Better multi-hop recall.
- Better answers to "what changed?" and "how is X related to Y?"
- Retrieval that reflects the existing store protocols instead of leaving them underused.

### Phase 6. Unify retrieval and push filters down

Mnemotree currently has two hybrid retrieval shapes and too much post-filtering.

Core changes:

- Standardize on one retriever path with provenance and explainability.
- Push filters into store queries wherever store capabilities allow it.
- Keep one ranking pipeline for vector, BM25, graph, and reranker signals.
- Return explanation metadata for why each memory was recalled.

Expected outcome:

- Lower complexity.
- Less accidental candidate loss.
- A retrieval system that is easier to debug and benchmark.

### Phase 7. Finish spaced repetition instead of only decaying scores

Mnemotree already has decay math and adaptive hooks. It needs the rest of the loop.

Core changes:

- Persist due-review schedules and replay state.
- Update stability based on successful recall events.
- Use retrievability when updating access metadata.
- Separate passive decay from active review and reinforcement.

Expected outcome:

- Real reinforcement behavior instead of importance-only drift.
- Better support for high-value persistent memories.
- A stronger science-based differentiator.

### Phase 8. Build the evaluation loop before expanding the surface area

Do not add all of the above without new benchmarks.

Add evaluation for:

- duplicate suppression
- correction accuracy
- contradiction handling
- temporal QA
- multi-hop graph recall
- consolidation quality
- p95 latency and cost

Expected outcome:

- The roadmap stays measurable.
- New memory features can compete on retrieval quality instead of marketing.

## Suggested Implementation Order

1. Phase 1 and Phase 2 first.
2. Phase 6 immediately after, to simplify the retrieval surface before adding more behavior.
3. Phase 3 and Phase 7 next, so truth and reinforcement become durable.
4. Phase 4 and Phase 5 after that, once the write and retrieval contracts are stable.
5. Phase 8 throughout, not at the end.

## What To Borrow

### From Engram

- Dream-cycle consolidation.
- Safety-critical memory classification.
- Temporal reasoning as a product-level feature.
- Better operational visibility into memory state.

### From Graphiti

- Episodic ingestion.
- Bi-temporal facts and relationship lifecycles.
- Hybrid graph + semantic retrieval.
- Temporal invalidation instead of silent overwrite.

### From Mem0

- Explicit add, update, and delete memory operations.
- Conflict resolution during ingestion.
- Graph context as a retrieval augmentation, not necessarily the top-level ranker.

### From Letta

- Pinned core memory blocks.
- Separate archival memory.
- Background sleeptime processing that updates long-lived memory.

## What To Avoid

- Do not copy SaaS or dashboard surface area before lifecycle correctness.
- Do not make graph storage mandatory for the core product.
- Do not make consolidation LLM-only.
- Do not introduce a new memory model so large that it breaks the current store contract unless there is a clear migration path.

## Sources

- Engram: https://github.com/heybeaux/engram
- Graphiti overview: https://help.getzep.com/graphiti/getting-started/overview
- Graphiti repo: https://github.com/getzep/graphiti
- Mem0 add memory: https://docs.mem0.ai/core-concepts/memory-operations/add
- Mem0 update memory: https://docs.mem0.ai/core-concepts/memory-operations/update
- Mem0 graph memory: https://docs.mem0.ai/open-source/features/graph-memory
- Letta memory blocks: https://docs.letta.com/guides/core-concepts/memory/memory-blocks/
- Letta archival memory: https://docs.letta.com/guides/core-concepts/memory/archival-memory/
- Letta sleeptime agents: https://docs.letta.com/guides/agents/architectures/sleeptime/
