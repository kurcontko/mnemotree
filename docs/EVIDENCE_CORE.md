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

## Configuration and embeddings

`Memory(path)` uses lexical retrieval. Pass an `Embedder` implementing
`embed_documents`, `embed_query`, and a stable `EmbeddingSpec` for dense/hybrid
retrieval. The database records provider, model, revision, and dimensions. A
different signature is rejected; rebuild into a fresh database when changing
models or adding embeddings to an existing lexical corpus. Reopening a vector
database without an embedder permits lexical reads, but rejects new ingestion.

Install `.[embeddings]` for the OpenAI-compatible HTTP provider, or `.[lite]` for
the local SentenceTransformer provider. The CLI accepts a JSON config:

```json
{
  "embedding": {
    "kind": "openai",
    "base_url": "https://your-provider.example/v1",
    "model": "your-embedding-model",
    "dimensions": 1024,
    "revision": "your-pinned-model-revision",
    "api_key_env": "EMBEDDING_API_KEY",
    "send_dimensions": true
  },
  "search": {
    "mode": "hybrid",
    "candidate_limit": 100,
    "limit": 10,
    "token_budget": 8000,
    "neighbor_chunks": 0
  }
}
```

Set `send_dimensions` false if your endpoint does not accept that parameter.
Use an explicit revision and rebuild when provider behavior changes; a revision
label cannot force an upstream API to pin weights. Never put keys in the config.
For local embeddings, set `kind` to `sentence-transformers` and supply `model`,
`revision`, and optional `query_prompt` / `document_prompt`. Model downloads are
explicit consequences of selecting that provider. No provider is contacted by
the default lexical configuration.

Python callers can provide a `Reranker` with async `score(query, texts)`, returning
one finite score per candidate. The baseline has no automatic reranker or LLM.
Use a fixed model and revision and record them separately when evaluating a
custom reranker; the report records its class, not arbitrary model internals.

## Storage and retrieval behavior

Each add is an ordered batch. Source position is within that request; session ID
groups batches. Message text is preserved exactly, and timestamp is an optional
timezone-aware source timestamp, separate from ingestion time. Explicit source
IDs must be unique per namespace; otherwise IDs derive from request and position.
Repeated text with different occurrence IDs remains distinct. The same request
and payload returns its receipt without calling an embedding provider again.
Different payloads using an existing request ID are conflicts.

Embedding calls complete before the transaction. Sources, chunks, FTS postings,
vectors, and the receipt commit together. `forget(namespace, source_id)` deletes
source chunks and both indexes. Receipts remain as tombstones: a retry cannot
silently resurrect a forgotten source. This is logical deletion, not a secure
erasure guarantee for SQLite files, WAL, or backups.

Chunks are fixed character spans within one message, with configurable overlap.
Retrieval filters namespace before each channel's candidate cap. FTS5 uses
sanitized OR query terms and BM25. Dense search uses exact cosine distance over
all eligible rows stored in sqlite-vec. The union is fused with weighted RRF,
optionally reranked, then packed. `limit` controls returned evidence count;
`candidate_limit` independently controls depth per retrieval channel.

`neighbor_chunks` expands a hit within its source message. The packer skips fully
contained duplicate spans and evidence that does not fit, then continues down
the ranking. It counts the exact rendered JSON evidence plus separators using
the selected tiktoken encoding. Partial overlapping spans can still repeat text.
Source ID, session, role, timestamp, and character offsets are inside each
rendered evidence block. Search does not update importance or access counters.

The first tokenizer use may download its public vocabulary. To prewarm a cache:

```bash
export TIKTOKEN_CACHE_DIR=/path/to/persistent/tokenizer-cache
python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
```

## Textual AML adapter

The optional service exposes `POST /v1/memories/add`, `POST /v1/memories/search`, and unauthenticated
`GET /health`. Add/Search require `Authorization: Bearer <MNEMOTREE_API_KEY>`.
It implements the [public textual API contract](https://agentmemoryleaderboard.ai/api-guide):
ordered user/assistant messages, optional Unix-millisecond timestamps, exact
ID echoes, synchronous durable acknowledgement, `user_id` isolation, optional
multiple-choice options, and ranked `{id, content}` results capped by `top_k`.
Multiple-choice options are appended to the retrieval query. Multimodal payloads
receive validation errors. No answer generation is performed.

The default evidence budget is 8,000 tokens. This counts returned content only;
the platform's final prompt also includes question, options, instructions, and
wrapping. The 117,760 input-token ceiling is not an evidence-only allowance.
Choose a budget with headroom and the correct answer tokenizer for a real run.
Smoke/deployment, model eligibility checks, and formal competition submission
remain separate steps. This branch has not been submitted or deployed.

## Reproducible retrieval evaluation

Use `mnemotree-evidence --db <fresh-path> --config <json> evaluate <corpus.jsonl>
<queries.jsonl> --output <report.json>`. The committed
[`examples/evidence`](../examples/evidence) files are tiny plumbing examples.

Each corpus line has `namespace`, `request_id`, `session_id`, and `messages`.
Each message uses the native `Message` schema: `content`, optional `id`, `role`,
and ISO timestamp. Supply explicit source IDs for labeled retrieval experiments.
Each query line has `id`, `namespace`, `query`, `relevant_source_ids`, and optional
`category`. Unknown gold source IDs and duplicate query IDs are errors.

Reports include corpus/query/code hashes, embedding specification, tokenizer,
SQLite/package versions, chunk/retrieval settings, per-query timings, candidate
source recall, packed source recall, all-source coverage, and reciprocal rank.
Empty-gold questions are counted separately and excluded from recall averages.
Source coverage is a coarse retrieval proxy: a partial chunk from a relevant
source can count as a hit even when it misses the answer-bearing span. It is
neither answer accuracy nor an abstention evaluation.

Compare lexical, dense, and hybrid with the same corpus, embedding model,
chunking, candidate depth, and token budget. Start each run with a fresh database.
First measure losses between candidate recall and packed recall; only then add
a reranker or another representation. Answer-model evaluation and official
dataset loaders are not included in this first iteration.

## Current limits and verification

Exact dense scoring is linear in eligible corpus size, not ANN. Fixed character
chunking can split sentences or code blocks. FTS BM25 statistics are global to
the database even though result eligibility is scoped. SQLite work is synchronous
and is intended for a modest local baseline; it can block an async event loop.
The parent baseline has no extraction or cross-message context expansion. The
stacked experiments add [atomic projections](ATOMIC_MEMORY.md) and
[neighboring-message context](CONVERSATION_CONTEXT.md), both optional. Temporal
contradiction resolution, automatic compaction, and legacy database migration
remain out of scope.

The tests exercise actual FTS5/sqlite-vec, reopen visibility, namespace isolation,
rollback after partial SQL writes, concurrent retries from separate connections,
deletion, malformed provider outputs, exact packing, HTTP contracts, and the
evaluation report. Deterministic test vectors do not demonstrate semantic quality.

```bash
pip install -e '.[dev,serve]'
pytest tests/test_evidence.py tests/test_evidence_adapter.py --no-cov
# The complete historical suite additionally needs the legacy extra.
pip install -e '.[dev,legacy]'
make lint typecheck test
```

## Work log

- Branch `feature/evidence-core` starts at main `f43ec23` in an isolated worktree.
- Existing dirty worktrees, including `feature/aml-lean`, are not modified.
- Source contracts/store, retrieval, and adapter/evaluation are committed in
  separate slices. The baseline is implemented; benchmark-quality measurement
  with real embedding models and answer evaluation remains future work.
