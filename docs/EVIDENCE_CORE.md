# Evidence core: first iteration

This branch replaces the default development path with a small, source-first
memory engine. The existing MemoryCore remains an optional legacy interface.

## Scope and acceptance

- Preserve ordered messages, roles, timestamps, stable occurrence IDs, and exact
  source spans. Identical text in different occurrences is not deduplicated.
- Commit sources, chunks, FTS5, vectors, and an idempotency receipt atomically.
  Acknowledged writes must be searchable after restart and from another instance.
- Require a namespace on every read and write. Apply eligibility before top-k.
- Use persistent SQLite FTS5 and sqlite-vec cosine search, with independently
  configurable candidate depth, RRF, optional reranking, and exact token packing.
- Keep search read-only. Report candidates, channel scores/ranks, packing losses,
  and timings; do not disguise retrieval recall as answer accuracy.
- Provide a runnable Add/Search adapter and a reproducible retrieval evaluation
  path. Exercise real FTS5/sqlite-vec locally, without paid model calls.
- Make the default import/install independent of LangChain, spaCy, NetworkX and
  scikit-learn. Preserve legacy integrations behind an explicit extra.

Deferred: automatic extraction, L1 facts, L2 scenes, L3 profiles, graphs,
reflection, decay, learned memory policies, and additional storage backends.
TencentDB's L0→L1→L2→L3 organization is a future experimental representation,
not a prerequisite for the baseline. No implementation is copied from Tencent,
Sen, or Swarm Brain; their source/projection separation informed the design.

Reference: https://github.com/TencentCloud/TencentDB-Agent-Memory

## Work log

- Branch `feature/evidence-core` starts at main `f43ec23` in an isolated worktree.
- Existing dirty worktrees, including `feature/aml-lean`, are not modified.
- Implementation and verification are in progress; this document is not a claim
  of benchmark quality or a successful competition submission.
