# Memory API

`mnemotree.Memory` is the library's public API: a few verbs over the Evidence
engine (`mnemotree.evidence.Memory`). It keeps what worked in the earlier
`MemoryCore` (agent-shaped verbs, small protocols as extension points, frozen
option objects) and drops its data model: there is one small record, and
anything optional will be metadata, not a field.

## Usage

```python
from mnemotree import Memory, Message, SearchOptions

with Memory("mem.sqlite", namespace="alice", embedder=None) as memory:
    item = await memory.remember("I moved to Lisbon", role="user", request_id="r1")
    turns = await memory.remember_many(
        [Message(content="Where should I live?", role="user"),
         Message(content="Lisbon has mild winters.", role="assistant")],
        session="chat-7",
    )
    recall = await memory.recall("where do I live", limit=5)
    recall.items      # MemoryItem records behind the packed evidence, rank order
    recall.context    # the packed text a reader sees
    recall.trace      # candidates, channel ranks, packing losses, timings
    await memory.get(item.id)
    await memory.forget(item.id)
```

| Verb | Contract |
| --- | --- |
| `Memory(path, *, namespace, embedder, reranker, token_counter, chunk_size, chunk_overlap, search)` | One namespace per instance; it is the isolation boundary, never a filter. Lexical without an embedder, hybrid with one. `search` fixes default `SearchOptions`. |
| `remember(content, *, role, session, timestamp, id, request_id)` | One durable, indexed message. Without `session` it is its own session, so it never becomes a neighbouring turn. Same `request_id` and body replays; a different body conflicts. |
| `remember_many(messages, *, session, request_id)` | Consecutive turns of one session, in order, as one atomic write. |
| `recall(query, *, limit, options)` | Read-only. Returns `Recall(items, evidence, context, trace)`. `items` are the sources of packed evidence, so a zero token budget returns none. |
| `get(id)` / `forget(id)` | Read one memory / delete it with its chunks and index entries. |

`MemoryItem` is `id, content, role, session, timestamp, created_at`. `content` is
the raw text exactly as stored.
