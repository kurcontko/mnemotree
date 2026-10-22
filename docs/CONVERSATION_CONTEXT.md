# Conversation-context experiment

Branch `experiment/conversation-context` is stacked on `experiment/atomic-memory`
at `e44d2d1`. It adds one optional retrieval behavior: neighboring original
messages around a retrieved source. It needs no additional LLM calls or tables.

The inspiration is Tencent's use of background conversation to interpret new
messages in its [L1 prompt formatter](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/6a4bb55ea54834d8265ce52024d3e3c75db30d38/MemoryCore/src/core/prompts/l1-extraction.ts).
Our adaptation uses context when returning retrieval evidence. This is an
independently implemented experiment, not a port of Tencent's retrieval code
or its L2 scene layer.

## Behavior

Set `neighbor_messages` to 1 or 2 in `SearchOptions` or CLI search configuration.
Zero is the library default; the AML service defaults to one.
`neighbor_chunks` still means adjacent chunks within the matching message;
`neighbor_messages` means adjacent messages within the same session.

Namespace and session must match. Order is acknowledged ingestion order within
the session, so context continues across Add requests: AML splits a session into
requests of at most 20 messages and sends them in source order. Concurrent Adds
to one session are ordered by commit, not payload. Deleted messages are not
restored; the next surviving message becomes the neighbor. (The first version
stopped at request boundaries, which dropped one side for every request's edge
turns and all context for single-message streaming Adds.)

Each neighbor is bounded by `neighbor_chars` (default 1,200). A preceding message
contributes its ending; a following message contributes its beginning. These
are exact spans, rendered as transcript lines around the anchor; their source
IDs and offsets are in the trace. They are not summaries.

The packer reserves room for ranked retrieval anchors first, then adds neighbor
spans if the complete rendered context stays within the token budget. Fully
contained duplicates are skipped, including neighbors already represented by
selected anchors. Partial overlaps can still repeat text. All expanded sources
are read in the same SQLite snapshot as the retrieval candidates. Traces expose
context candidates and spans omitted through budget or duplicate suppression.

## Isolated ablation

```bash
mnemotree-evidence --db context-experiment.sqlite \
  --config examples/evidence/context.json \
  ablate-context examples/evidence/context-corpus.jsonl \
  examples/evidence/context-queries.jsonl --output context-ablation.json
```

The database must be fresh. The command ingests once and compares context off
against the configured neighbor count (or one if the count is zero). Retrieval
representation, candidate depth, and token budget remain fixed. It works with
raw, atom, or combined retrieval; optional extraction runs once per experiment.

Reports distinguish candidate source recall, recall after possible context
expansion, packed anchor recall, and total packed source recall. Expansion can
discover a gold source without improving anchor retrieval. Scores remain coarse
source-coverage proxies; real answer accuracy must be evaluated separately.

The tiny committed example demonstrates a query matching the assistant's
question while its adjacent user answer lacks the query terms. It is a plumbing
example, not a benchmark or evidence that this feature should become a default.

## Stack and boundaries

```
feature/evidence-core
  experiment/atomic-memory
    experiment/conversation-context
```

The parent branches remain independently runnable. There is no L2/L3 pyramid,
automatic scene generation, profile consolidation, decay, or mutable truth
management. The experiment is intentionally small enough to remove if the
extra context costs more tokens than it improves answers.
