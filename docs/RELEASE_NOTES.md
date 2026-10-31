# 1.0.0 (2026-10-31)

Mnemotree now stores original messages in SQLite and retrieves source evidence.
The public `Memory` API binds one namespace and provides `remember`,
`remember_many`, `recall`, `get` and `forget`. The same verbs are available through
the standard-library MCP stdio server.

## Upgrading from 0.4.0 or earlier

`MemoryCore`, its pluggable vector/graph backends, framework adapters and the older
MCP server have been removed. Their databases cannot be opened by `Memory`, and
there is no automatic import. Keep existing databases and export original source
messages before moving to a fresh database through `remember_many` or the native
JSONL ingestion CLI. Review source IDs, roles and timestamps during that transfer.

Python 3.11 or later is required. The default installation uses Pydantic,
sqlite-vec and tiktoken; HTTP providers, the local embedding model and the HTTP
service are optional extras. See the [guide](GUIDE.md) for their configuration.

## Changes from earlier 0.5 development snapshots

- MCP `recall.memories` contains source metadata without full `content`. Fetch
  complete messages through `get`.
- `Recall.items` includes neighboring source messages and shares the evidence's
  read snapshot, even when another instance changes the database during a search.
- `trace.packed_tokens` counts the complete joined context, including separators.
  `trace.evidence_tokens` counts separately rendered blocks. Both fit the evidence
  budget; metadata and JSON framing are additional response overhead.
- Source timestamps retain their supplied offsets, so relative-date notes use the
  source's local calendar day. Earlier snapshots normalized timestamps to UTC;
  their original offsets cannot be recovered from existing records. Reingest
  original messages into a fresh database if those offsets are needed.
- A timestamp's offset is part of a request's payload identity. Replaying a
  `request_id` with the same instant at a different offset raises `IngestConflict`,
  so a retry should resend the original timestamp unchanged.
- HTTP embeddings require Tenacity 9.2.1 or later, matching the retry API in use.
- Distributions include `py.typed`, and the development extra includes `build`.
