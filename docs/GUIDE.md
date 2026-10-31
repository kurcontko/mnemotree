# Mnemotree guide

Mnemotree stores original messages in SQLite and retrieves attributed source spans
through persistent FTS5 and sqlite-vec indexes, hybrid rank fusion, optional
reranking and exact token-budget packing. This guide covers the `Memory` API, the
MCP server, configuration, retrieval behavior, the optional atomic projections, the
AML Add/Search service, evaluation and current limits.

## Memory API

`mnemotree.Memory` binds one namespace over the engine (`mnemotree.evidence.Memory`,
which takes a namespace on every call and exposes every retrieval option).

```python
from mnemotree import Memory, Message

with Memory("mem.sqlite", namespace="alice", embedder=None) as memory:
    item = await memory.remember("I moved to Lisbon", role="user", request_id="r1")
    turns = await memory.remember_many(
        [Message(content="Where should I live?", role="user"),
         Message(content="Lisbon has mild winters.", role="assistant")],
        session="chat-7",
    )
    recall = await memory.recall("Lisbon", limit=5)  # lexical matching without an embedder
    recall.items      # MemoryItem source snapshots, including neighboring turns
    recall.context    # the packed text a reader sees
    recall.trace      # candidates, channel ranks, packing losses, timings
    await memory.get(item.id)
    await memory.forget(item.id)
```

| Verb | Contract |
| --- | --- |
| `Memory(path, *, namespace, embedder, reranker, token_counter, chunk_size, chunk_overlap, search)` | One namespace per instance; it is the isolation boundary, never a filter. Lexical without an embedder, hybrid with one. `search` fixes default `SearchOptions`. |
| `Memory.from_engine(engine, *, namespace, search)` | Bind an engine that is already configured, such as one built from a CLI config. Closing the `Memory` closes the engine. |
| `remember(content, *, role, session, timestamp, id, request_id)` | One durable, indexed message. Without `session` it is its own session, so it never becomes a neighbouring turn. Same `request_id` and body replays; a different body conflicts. |
| `remember_many(messages, *, session, request_id)` | Consecutive turns of one session, in order, as one atomic write. |
| `recall(query, *, limit, options)` | Read-only. Returns `Recall(items, evidence, context, trace)`. `items` are distinct source snapshots of packed evidence, including neighboring turns; a zero token budget returns none. `limit` caps evidence blocks, which can contain multiple sources. |
| `get(id)` / `forget(id)` | Read one memory / delete it with its chunks and index entries. |

`MemoryItem` is `id, content, role, session, timestamp, created_at`. `content` is the
raw text exactly as stored. Sources and evidence come from the same read snapshot,
even if another process replaces or deletes a source during retrieval. The source
order follows evidence blocks, with each anchor followed by its neighbors; a source
appears only once. `timestamp` preserves the supplied timezone offset.

## MCP server

`mnemotree-mcp` serves one namespace over stdio. It uses only the standard library
(newline-delimited JSON-RPC 2.0, protocol versions 2024-11-05 to 2025-11-25), so a
bare install runs it. Tools: `remember`, `remember_many`, `recall`, `get`, `forget`.
Results carry JSON text and, from 2025-06-18, `structuredContent`; argument and
conflict errors are tool results with `isError`, so the model can correct itself.

`recall` returns `context`, `packed_tokens` and `memories`. Here `memories` contains
source metadata (ID, role, session and timestamps), without full message content.
Use `get` to fetch full messages by ID. `packed_tokens` counts the context only;
JSON framing and source metadata add response overhead.

```bash
claude mcp add mnemotree -- mnemotree-mcp --db ~/.mnemotree/memory.sqlite \
  --namespace "$USER" --config ~/.mnemotree/config.json   # or --lexical
```

`--db`, `--namespace` and `--config` fall back to `MNEMOTREE_MCP_DB`,
`MNEMOTREE_MCP_NAMESPACE` and `MNEMOTREE_MCP_CONFIG`. `--config` is the same JSON
file `mnemotree-evidence` reads ([Configuration](#configuration)). A config without
an embedding is refused unless `--lexical` is given: keyword-only retrieval misses
paraphrases, so it is a choice, never a fallback.

Each client session starts its own process. Processes pointed at the same `--db`
share it: SQLite WAL serializes writes, and every acknowledged write is visible to
the next recall in any session. Concurrent first opens of a new file retry briefly.
Separate processes do not share an embedder, though: each loads its own, which
matters for the local `lite` model.

Stored text is plain text in the database and comes back verbatim, so anything
stored can reach the model: keep secrets out of it. Text that a model saved from
an untrusted page can also carry instructions into later sessions. The server
tells clients to read memories as records, not instructions, but only the client
can enforce that. Approval is the client's job too: `forget` is marked
`destructiveHint` so clients can ask before deleting, and deletion is permanent.

## Configuration

`Memory(path)` uses lexical retrieval. Pass an `Embedder` implementing
`embed_documents`, `embed_query` and a stable `EmbeddingSpec` for dense/hybrid
retrieval. The database records provider, model, revision and dimensions, and
rejects a different signature: rebuild into a fresh database when changing models
or adding embeddings to a lexical corpus. Reopening a vector database without an
embedder permits lexical reads but rejects new ingestion.

Install `mnemotree[embeddings]` for the OpenAI-compatible HTTP provider, or
`mnemotree[lite]` for the local SentenceTransformer provider. The CLIs accept a JSON
config:

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

Keys never go in the config: `api_key_env` names the environment variable that holds
them (`EMBEDDING_API_KEY` by default). Set `send_dimensions` to false if the endpoint
does not accept that parameter. A revision label cannot force an upstream API to pin
weights, so use an explicit revision and rebuild when provider behavior changes. For
local embeddings, set `kind` to `sentence-transformers` and supply `model`,
`revision` and optional `query_prompt` / `document_prompt`; selecting it downloads
the model. The default lexical configuration contacts no provider.

Python callers can pass a `Reranker` with async `score(query, texts)` returning one
finite score per candidate. There is no automatic reranker or LLM. Evaluation
reports record a custom reranker's class, not its model, so record the model and
revision separately.

Token counting uses tiktoken's `o200k_base` by default (`tokenizer` in the CLI
config), which downloads its public vocabulary on first use. To run offline, prewarm
a persistent cache:

```bash
export TIKTOKEN_CACHE_DIR=/path/to/persistent/tokenizer-cache
python -c 'import tiktoken; tiktoken.get_encoding("o200k_base")'
```

## Storage and retrieval

Every read and write requires an exact namespace; search spans all sessions within
it, and eligibility is applied before top-k.

**Ingestion.** Each add is an ordered batch: source position is within the request,
and the session ID groups batches. Message text is preserved exactly; identical text
in different occurrences is not deduplicated. The optional timestamp is a
timezone-aware source time, separate from ingestion time; its offset is retained.
Explicit source IDs must be unique per namespace; otherwise IDs derive from request
and position. The same
request and payload returns its receipt without calling the embedding provider
again; a different payload under an existing request ID is a conflict.

Embedding calls complete before the transaction. Sources, chunks, FTS postings,
vectors and the receipt commit together, so an acknowledged write is searchable after
a restart and from another instance. `forget` deletes the source's chunks and both
indexes. Receipts remain as tombstones, so a retry cannot resurrect a forgotten
source. This is logical deletion, not secure erasure from SQLite files, WAL or
backups.

**Retrieval.** Chunks are fixed character spans within one message, with
configurable overlap. The namespace filter runs before each channel's candidate cap.
FTS5 uses sanitized OR query terms and BM25; dense search uses exact cosine distance
over every eligible row in sqlite-vec. The union is fused with weighted RRF,
optionally reranked, then packed. `candidate_limit` sets the depth of each channel;
`limit` sets how much evidence is returned. Search is read-only: it updates no
importance or access counters.

**Packing.** `neighbor_chunks` expands a hit within its source message. The packer
skips fully contained duplicate spans and evidence that does not fit, then continues
down the ranking. `trace.packed_tokens` counts the exact `context` string, including
blank-line separators, using the selected tokenizer. `trace.evidence_tokens` is
the sum of separately rendered blocks, as delivered by the AML service. Both counts
must fit `token_budget`. JSON framing, source records and trace metadata are outside
that evidence budget. Partially overlapping spans can still repeat text.

Context is a plain transcript, for example `user: I grow orchids.` A timestamped
message includes its date, weekday, time and UTC offset. Recognized relative dates
also receive rule-based notes, resolved against the source's own calendar day:

```text
[2026-10-30 Fri 00:30 UTC+02:00] user: I adopted a cat yesterday. (yesterday = Thursday, 29 October 2026)
```

These notes are derived hints, not part of the stored text. Source IDs, sessions,
roles, timestamps and character offsets are available on `evidence`; the trace adds
candidates, channel scores and ranks, packing losses and timings. The engine's
`SearchResult.sources` contains full source snapshots for Python callers and is
excluded from JSON serialization.

## Atomic projections

An optional `Extractor` turns ordered passages into standalone statements used only
for retrieval; answer context uses source spans rather than generated statements.
Transcript headers and date notes follow the rendering rules above. Raw-only
retrieval stays the default. The design follows Tencent's standalone L1 statements
and source-link attribution in TencentDB-Agent-Memory at commit
`6a4bb55ea54834d8265ce52024d3e3c75db30d38`
([L1 prompt](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/6a4bb55ea54834d8265ce52024d3e3c75db30d38/MemoryCore/src/core/prompts/l1-extraction.ts),
[L1 extraction pipeline](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/6a4bb55ea54834d8265ce52024d3e3c75db30d38/MemoryCore/src/core/record/l1-extractor.ts)).
The Python implementation and prompt are written independently; no Tencent code or
prompt text is copied. Scenes, importance thresholds, personas, deduplication,
merging and profiles are not implemented.

**Ingestion.** The extractor returns `AtomDraft(text, citations)`; each citation names
a passage ID and an exact, unambiguous source quote. The engine resolves quotes to
character offsets and rejects missing, ambiguous, blank or out-of-batch citations.
A valid quote proves provenance, not that the statement follows from it. Calls are
bounded by `extraction_batch_chars` of source text (default 12,000, a character cap,
not a token guarantee); long messages are split losslessly, references cannot be
resolved across batches, and at most 64 atoms per call are accepted. Repeated
occurrences stay separate and conflicting statements are kept.

The database pins provider, model, revision, prompt hash, batching and pipeline
version; changing any of them requires a fresh database. Reopening without an
extractor permits reads but rejects writes to an extracted corpus. Sources, raw
indexes, atoms, citation spans, atom indexes and the receipt commit in one
transaction, and an extraction or embedding failure aborts the add. Deleting any
cited source invalidates the whole dependent atom. An empty extraction is valid and
still stores the raw messages.

**Retrieval.** `SearchOptions.representation` is `raw` (default), `atoms` or
`combined`. `atoms` searches only the derived indexes, then retrieves the original
chunks overlapping their citations; `combined` adds those candidates to the raw
channels before RRF, scaled by `atom_weight`. `candidate_limit` caps each channel
before citation expansion; returned evidence and the token budget stay bounded. The
trace includes atom text and source spans and may contain sensitive material; AML
responses never include it.

The extractor is an **engine** option. Configure `examples/evidence/atomic.json`
(key variable `EXTRACTION_API_KEY`), or bind a configured engine to the public API:

```python
import os
from mnemotree import Memory, SearchOptions
from mnemotree.evidence import Memory as EvidenceMemory
from mnemotree.evidence.atoms import OpenAIExtractor

engine = EvidenceMemory(
    "atoms.sqlite",
    extractor=OpenAIExtractor(
        model="your-extraction-model",
        revision="your-pinned-model-revision",
        api_key=os.environ["EXTRACTION_API_KEY"],
    ),
)
with Memory.from_engine(
    engine, namespace="alice",
    search=SearchOptions(mode="lexical", representation="combined"),
) as memory:
    item = await memory.remember("I grow orchids on my balcony.")
    recall = await memory.recall("orchids")
```

This example needs the `embeddings` extra and runs inside an async function or a
notebook with top-level `await`, like the API example above.

## AML Add/Search service

`mnemotree-evidence serve` (the `serve` extra) implements the
[textual Agent Memory Leaderboard API](https://agentmemoryleaderboard.ai/api-guide):
`POST /v1/memories/add`, `POST /v1/memories/search` and unauthenticated `GET /health`.
Add and Search require the key from `MNEMOTREE_API_KEY` (or `--key-env`), sent as
`Authorization: Bearer <key>`, `Authorization: Token <key>` or `X-Api-Key: <key>`. It
accepts ordered user/assistant messages with optional Unix-millisecond timestamps,
echoes IDs exactly, acknowledges writes durably and synchronously, isolates by
`user_id`, appends multiple-choice options to the retrieval query, and returns ranked
`{id, content, score}` results capped by `top_k`, with `created_at` on dated evidence.
Multimodal payloads are rejected, and no answer is generated.

```bash
pip install 'mnemotree[serve]'
export MNEMOTREE_API_KEY='replace-with-your-service-key'
mnemotree-evidence --db service.sqlite serve
```

The default evidence budget is 8,000 tokens of returned content. The platform's
final prompt also holds the question, options and instructions, so the 117,760
input-token ceiling is not an evidence allowance: leave headroom and count with the
answer model's tokenizer. `--request-log` (`AML_REQUEST_LOG`) appends one
metadata-only line per request; `--sample-payloads` (`AML_SAMPLE_PAYLOADS`) and
`--sample-limit` (`AML_SAMPLE_LIMIT`, default 50) sample payloads for smoke
inspection.

Without a `search` block, the service applies its defaults (`aml_options` in
`evidence/server.py`): hybrid retrieval when an embedder is configured, one neighboring
message merged into each hit, one chunk per source message and at most 100 items. Each
channel returns 100 candidates, or 300 when a history averages more than 2 chunks per
message. User turns rank first when assistant text is at least 65% of a history. A
`search` block replaces these defaults rather than merging with them.

`docker/aml/` holds a Caddy-fronted deployment and its smoke test. Its
`config.example.json` uses `text-embedding-v4` at 1,024 dimensions through DashScope's
OpenAI-compatible endpoint, with no LLM or reranker. On a host with ports 80 and 443
open and a DNS A record:

```bash
cd docker/aml
cp .env.example .env  # fill in the keys, AML_DOMAIN and a fresh AML_DB
docker compose --profile caddy up -d --build
set -a; . ./.env; set +a
./smoke.sh "https://$AML_DOMAIN"
```

## Evaluation

```bash
mnemotree-evidence --db <fresh.sqlite> --config <config.json> \
  evaluate <corpus.jsonl> <queries.jsonl> --output <report.json>
```

Each corpus line has `namespace`, `request_id`, `session_id` and `messages` in the
native `Message` schema (`content`, optional `id`, `role`, ISO timestamp); give
explicit source IDs for labelled experiments. Each query line has `id`, `namespace`,
`query`, `relevant_source_ids` and optional `category`. Unknown gold IDs and duplicate
query IDs are errors. [`examples/evidence`](../examples/evidence) holds tiny plumbing
examples, not evidence of quality.

Reports include corpus, query and code hashes, the embedding spec, tokenizer,
SQLite and package versions, chunk and retrieval settings, per-query timings,
candidate and packed source recall, all-source coverage and reciprocal rank.
Empty-gold queries are counted separately and excluded from recall averages. Source
coverage is a coarse retrieval proxy, not answer accuracy: a partial chunk of a
relevant source counts as a hit even when it misses the answer-bearing span.

Ablations ingest once into a fresh database and compare variants on the same corpus,
tokenizer and budget:

- `ablate`: raw, atoms and combined representations. Reports shared Add latency, atom
  count, extractor configuration and token usage when the provider reports it. For the
  ingestion-cost control, run `evaluate` without extraction on another fresh database.
- `ablate-context`: neighbouring messages off and on.
- `ablate-retrieval`: retrieval, role, query, context and budget policies.

Compare lexical, dense and hybrid with the same corpus, model, chunking, candidate
depth and budget. Measure the loss between candidate and packed recall before adding
a reranker or another representation.

## Limits

- Dense scoring is exact, so it is linear in eligible corpus size; there is no ANN.
- Fixed character chunking can split sentences or code blocks.
- FTS BM25 statistics are global to the database, though eligibility is scoped.
- SQLite work is synchronous and can block an async event loop; the target is a
  modest local store.
- No temporal contradiction resolution, automatic compaction, or import of
  databases from releases up to 0.4.0.
