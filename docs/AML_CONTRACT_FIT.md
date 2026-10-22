# AML contract fit

Branch `experiment/aml-contract-fit` is stacked on
`experiment/conversation-context` at `151f412`. It aligns the evidence service
with the published AML textual contract and fixes retrieval gaps found while
reviewing the stack.

## Contract facts that drove the changes

- Formal Search sends `top_k=100`. The answer model receives a 117,760-token
  prefix of the returned evidence, in returned order, including instructions.
- Add carries at most 20 messages or 2,000 words; a session spans many Adds.
  Streaming interleaves Add and Search.
- Responses may carry `score` and `created_at`; some answer prompts render
  memories as `- [created_at] text`.
- 400/401/403/404/422 are never retried and stop the stage.

## Changes

| Area | Before | Now |
|---|---|---|
| Service defaults | 10 results, 8,000 tokens, no context | 100 results, 8,000 tokens, ±1 message, merged blocks (`aml_options`) |
| Token counting | `cl100k_base` | `o200k_base` (gpt-4o family) |
| Rendering | JSON with IDs, spans, offsets | `[2024-01-01 Mon 00:00 UTC] user: text` lines, neighbors in order |
| Response | `{id, content}` | `{id, content, score, created_at?}` |
| Neighbor scope | Same Add request | Same session, ingestion order |
| Lexical index | `unicode61`, CJK runs as one token, all query terms | Porter `unicode61`, CJK bigrams, query stopwords removed |
| Blank turns | 422 for the whole Add | Skipped |
| Embedding batches | Sequential | Concurrent (`max_concurrency`, default 8) |
| Returned items | One per anchor | Optional `merge_blocks`: anchors within `1 + 2 * neighbor_messages` messages of each other in a session form one item |

With `merge_blocks`, `limit` (AML `top_k`) counts blocks rather than turns. A
block keeps its best anchor's rank, ID, and score; its content is one ordered
transcript in which overlapping spans of a message are joined and `...` marks
skipped messages. At 8K on LoCoMo the 100-item cap, not the budget, was the
binding constraint (contexts averaged 4.7K tokens). Measured on full LoCoMo
(1,540 questions, 8K, hybrid, Qwen3-Embedding-0.6B):

| Config | All evidence | Loss: retrieval / packing | Tokens |
|---|---|---|---|
| no merge, ±0 | 72.7% | 21.7% / 5.6% | 4,681 |
| merge, ±0 | 77.9% | 21.7% / 0.4% | 5,859 |
| merge, ±1 | **86.7%** | 13.3% / 0.0% | 7,992 |

Merging is on in `aml_options` and off in the library default. The packer
re-counts only the item it changes, which cut packing from about 500 ms to
under 20 ms for merged ±1 searches. Sources store a per-session sequence number, since rowids
interleave sessions.

Schema version is now 4: version 2 changed the lexical index, version 3
partitions both vector tables by namespace, and version 4 scopes the full-text
indexes by namespace. Older databases are rejected and must be rebuilt.

Dense search used to join a namespace's chunks to `vec0` row by row, about
0.4 ms per chunk, so search time grew with each user's history: 1.0 s on a
500K-token BEAM conversation, and by extrapolation tens of seconds at 10M.
A `vec0` KNN query restricted to the namespace partition returned the same
top 100 in 5 ms.

Lexical search had the same shape at corpus level: FTS5 scored every matching
row in the database and filtered to the namespace afterwards, so its cost
followed the total corpus. Each row now also indexes a digits-only namespace
token (digits pass the Porter stemmer unchanged), and queries match
`scope : "<token>" AND text : (...)` with BM25 weight 0 on the scope column. The
exact namespace join stays as the guarantee. On the 572K-chunk BEAM 10M store
this took lexical search from 209 ms to 69 ms. Rebuilt LoCoMo-Refined and BEAM
100K stores retrieved the same evidence (88.8% vs 89.0%, and 42.3% for both),
and LoCoMo-Refined search p50 fell from 336 ms to 131 ms with both changes.

## Measured defaults

On the AML public-prompt harness, with qwen3.6-35b-a3b answering and judging and
hybrid search, merged ±1 blocks at 8K tokens scored:

| Dataset | Questions | Plain 8K | Merged ±1 8K | Merged ±1 16K |
|---|---|---|---|---|
| LoCoMo | 396 | 43.2 | 51.8 | 52.0 |
| BEAM 100K | 400 | 43.1 | 45.0 | 45.0 |
| ScriptMem (Enemy of the People) | 94 | 69.1 | 72.3 | 72.3 |
| PersonaMem v2 | 280 | 37.1 | 36.8 | 37.9 |

LongMemEval-S stayed within 62-67 for every configuration. The service budget is
therefore 8,000 tokens rather than most of AML's 117,760-token window. The
platform's reader is gpt-4o-mini, not qwen3.6, so a run on the
platform's own models remains the check.

Still open: ±1 vs ±2 neighbors, stopword removal on non-English queries, and
extractor robustness (reasoning-model output, one malformed reply failing an Add).

## Relative dates

Rendered turns end with rule-resolved notes for relative phrases, anchored on the
message's own timestamp: "I went yesterday. (yesterday = Sunday, 7 May 2023)".
`evidence/temporal.py` covers days, weeks, months, years, "N units ago" and
weekdays; a bare "on Saturday" resolves forward only when its sentence is
future-tense. No LLM is involved. Notes are added at render time, so the
indexed text and lexical matching are unchanged. In a peer harness on LoCoMo with
qwen3.6, the same rules gave +2.5 overall and +13 to +16 on temporal questions;
LongMemEval-S, whose temporal questions are interval arithmetic, did not move.
