# Benchmarks

In these runs Mnemotree retrieves evidence from a long conversation history, a
language model answers from it, and a judge scores the answer. The README shows the
headline numbers. This page records how we produced them and what they leave out.

## Setup

| | |
| --- | --- |
| Engine | `v1.0.0` (`02aefee`) |
| Retrieval | Hybrid FTS5 BM25 and sqlite-vec, fused with RRF, no reranker |
| Evidence budget | 8,000 tokens of item content per question |
| Embeddings | DashScope `text-embedding-v4`, native API, 1,024 dimensions |
| Reader and judge | `openai/gpt-4o-mini` through OpenRouter, OpenAI route, no fallbacks, temperature 0 |
| Stores | Fresh stores, one namespace per conversation history |

The AML Agent Memory Leaderboard prescribes these models for its open-source track.
We ingest each history in order, with its original roles and timestamps. Only the
scorer sees gold answers and evidence labels. The run behind these numbers used
engine `e9e694c`, whose `src/` tree is identical to `02aefee`.

## Method

The benchmarks use the default profile of the AML Add/Search service,
`aml_options` in `src/mnemotree/evidence/server.py`. No LLM, reranker or extractor
runs inside Mnemotree. The embedder is the only model it calls.

- **Storage.** Mnemotree stores each message verbatim in SQLite and splits it into
  1,200-character chunks that overlap by 200 characters. Every chunk keeps its
  source message, session, role and timestamp.
- **Retrieval.** FTS5 BM25 and exact sqlite-vec cosine search over
  `text-embedding-v4` both run inside the namespace. Reciprocal rank fusion merges
  their rankings with k = 60 and equal weights. For multiple-choice questions the
  options go into the query. The embedder sees at most 6,000 tokens of a query.
- **Candidates.** Each channel returns 100 candidates. When a history averages more
  than two chunks per message it returns 300, so long assistant turns don't crowd
  out the pool. Only the best chunk of each message competes for a ranked slot.
- **Ordering.** When assistant text makes up at least 65% of a history, user turns
  rank first. Otherwise the fused order stands.
- **Context.** Hits go into the pack in rank order until it reaches 8,000 tokens or
  100 items. Hits up to three messages apart in one session merge into a single
  passage. Each hit can then bring in one neighboring message on each side, from
  the same session and cut to 1,200 characters, but only into budget the hits left
  free. The other chunks of a hit's message go in last, if room remains.
- **Dates.** The pack shows each timestamped message with its date, weekday, time
  and UTC offset. Rule-based notes resolve relative dates such as "yesterday"
  against the message's own timestamp.

We picked the 8K budget, the neighbor window, merging, one chunk per message,
user-first ordering and the density-gated candidate pool from local runs over
LoCoMo, LongMemEval-S, BEAM, ScriptMem and PersonaMem.

## Evidence budget

The budget is a setting, and `token_budget` accepts up to 117,760 tokens. Changing
it on default `SearchOptions` won't reproduce the benchmarks, though. The defaults
have no neighbor window, no merging, no per-message chunk cap and no user-first
ordering. To keep the benchmarked retrieval and change only the budget, start from
the values `aml_options` sets in `src/mnemotree/evidence/server.py`:

```python
from mnemotree import Memory, SearchOptions

BENCHMARK = SearchOptions(
    limit=100,
    neighbor_messages=1,
    merge_blocks=True,
    chunks_per_source=1,
    role_first="user",
    role_first_min_assistant_share=0.65,
    deep_candidate_limit=300,
)

memory = Memory(
    "memory.sqlite",
    namespace="alice",
    embedder=embedder,
    search=BENCHMARK.model_copy(update={"token_budget": 24_000}),
)
recall = await memory.recall("Where did I move last year?")
wide = await memory.recall(
    "Where did I move last year?",
    options=BENCHMARK.model_copy(update={"token_budget": 48_000}),
)
```

Packing stops at whichever runs out first, the token budget or `limit`. The profile
already sets `limit` to 100, the most a search returns. If you build options of
your own, raise `limit` along with the budget.

The service reads the same fields from the `search` block of its config. A `search`
block replaces the default profile instead of merging with it, so it needs the
fields above to keep the benchmarked behavior. The service ignores `limit` there.
Each request's `top_k` sets it, capped at 100. `trace.packed_tokens` reports how
many tokens the pack used, counted with the configured tokenizer.

Pick the budget for the reader. In our tuning runs gpt-4o-mini answered better from
8K than from 24K tokens on most sets, while Qwen3.6-35B-A3B, a larger local reader,
did better at 24K. Extra evidence only helps a model that can use it. Size the
budget to the answer model and leave room for its instructions and the answer.

## Results

| Benchmark | Questions | Metric | Score |
| --- | ---: | --- | ---: |
| LoCoMo | 1,540 | Judge accuracy over categories 1 to 4, adversarial category 5 excluded | 72.1 |
| LongMemEval-S | 500 | Judge accuracy, cleaned release, all questions | 60.8 |

### LoCoMo-Refined, official scorer

A separate run on `02aefee` used the unmodified
[LoCoMo-Refined evaluator](https://github.com/mem-eval-suite/LoCoMo_refined) at
`8870911`, which fixes Qwen3-14B as the judge. gpt-4o-mini answered. Embeddings came
from Nemotron-3-Embed-8B in BF16 at 4,096 dimensions. LoCoMo-Refined is a cleaned
release of LoCoMo with 1,382 questions, against 1,540 in categories 1 to 4 of the
original.

| Category | Questions | Judge score |
| --- | ---: | ---: |
| 1 | 213 | 65.7 |
| 2 | 299 | 63.2 |
| 3 | 68 | 52.9 |
| 4 | 802 | 90.2 |
| **Overall** | **1,382** | **78.7** |

We ran the official scorer ourselves. The organizers have not reviewed or placed
this result.

## Not reported yet

- **BEAM.** Published BEAM results use other answer models and judges, so we have
  nothing comparable to set our number against. We will publish BEAM after a run
  that matches a published setup.
- **LongMemEval-V2.** Its histories are web and enterprise agent trajectories made
  of ordered states, actions, accessibility trees and screenshots. Mnemotree is
  built for conversational text and handles that data poorly. Tuning won't fix it.
  It needs engine changes.

## References

- Reciprocal rank fusion: Cormack, Clarke and Büttcher, SIGIR 2009.
- BM25 through SQLite FTS5: Robertson et al., Okapi; Robertson and Zaragoza, 2009.
  The tokenizer uses the Porter stemmer from Porter 1980, English stopwords close
  to the NLTK and Snowball lists, and overlapping CJK bigrams.
- [sqlite-vec](https://github.com/asg017/sqlite-vec) by Alex Garcia for exact vector
  search. Alibaba `text-embedding-v4`, from the Qwen3-Embedding series,
  arXiv:2506.05176. tiktoken for token counting.
- The background conversation in TencentDB-Agent-Memory's L1 prompts, at commit
  `6a4bb55`, gave us the idea for the neighbor window and merged passages. Our
  implementation is independent.

## Limitations

- **One draw per score.** Re-answering identical contexts flips a few percent of
  judgments, which moves a score by one to two points. Treat differences under
  about two points as noise.
- **Published rows are not reruns.** The README's reference numbers come from the
  Mem0 and Zep papers, with their own prompts, judges and context sizes. The Zep
  paper judges LongMemEval with gpt-4o.
- **No question date on LongMemEval.** Our reader prompt leaves out the date the
  question is asked, so temporal-reasoning questions are harder for our reader than
  in the official setup.
- **Harness not public.** The drivers that produced these numbers are not in this
  repository yet, so nobody can reproduce the results from it alone.
