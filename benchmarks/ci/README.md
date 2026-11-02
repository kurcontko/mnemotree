# Frozen memory evaluations

CI rebuilds memory from real LoCoMo and LongMemEval-S histories and runs the
current checkout's public `Memory.remember_many` and `Memory.recall` methods.
Embedding responses come from a completed local **Qwen3-Embedding-0.6B** run.
The replay disables network access, including tokenizer downloads. The runner
needs only the library's core dependencies; no model server, GPU, credentials,
or reader/judge calls are required.

```sh
PYTHONPATH=src python .github/scripts/check_memory_evals.py --output /tmp/memory-evals
```

The `memory_evals` job joins CI's required `gate` on code changes. It writes a
job summary and uploads the JSON comparison, including failing question IDs,
evidence coverage, source IDs, and context hashes.

## Frozen inputs

`fixtures/open-model-v1.zip` contains 42 questions across 14 complete histories:

| Dataset | Questions | Histories | Selection |
|---|---:|---:|---|
| Original LoCoMo | 32 | 4 | Two questions per category in each history; adversarial category excluded |
| LongMemEval-S cleaned | 10 | 10 | One question per category, including the four abstention categories |

Selection sorts IDs by `SHA256("mnemotree-ci-v1:" + id)`, independently of
retrieval or answer outcomes. All distractors in each selected history are
retained. Roles, timestamps, session order, repeated source occurrences, and
image captions follow the recorded October 27 dataset projection. Original
LoCoMo speaker names are not newly prefixed to messages. Gold answers are not
included. Evidence labels are separate from the ingestion/question projections
and are used only after retrieval.

Vectors were recorded with
[`Qwen/Qwen3-Embedding-0.6B@97b0c61`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md)
(Apache-2.0). The fixture uses its first 256 coordinates, L2-normalized and
stored as little-endian float32. This model supports Matryoshka output dimensions
from 32 to 1024. The projection keeps the fixture small; its new retrieval
baseline is independent of the old 1024-dimensional reader scores. The original
query instruction, truncation policy, cache identity, source hashes, and
selection are recorded in the archive's `provenance.json`. The frozen
`o200k_base` vocabulary accounts for the engine's existing 8K token budget; it
does not supply inference outputs.

`fixture.json` pins the complete archive checksum; the internal provenance pins
every member. Exact text and query/document type identify each vector. Missing
vectors fail with **cache refresh required**. Neither lexical fallback nor
automatic model calls can fill a miss.

## Gate semantics

Every question checks hybrid retrieval, candidate visibility, exact source
spans, original text/roles, returned item limits, and packed token accounting.
The database contains multiple benchmark namespaces. Each question ingests only
its allowed prefix before recall.

`baseline.json` records per-case floors for candidate turn recall, packed turn
recall, complete annotated evidence coverage, and packed session recall. A
decrease on any labeled case fails. Different contexts with preserved or improved
coverage pass and are listed in the report. These strict floors are deliberate
regression checks on this small development fixture, not significance tests or
estimates of full-benchmark quality. Intentional tradeoffs require explicit
baseline review.

The initial baseline was captured against main at
`02aefee00eed5c8244e84b3fea13b9ef6728e823`. Both the joined context (including
separators) and separately rendered evidence blocks must fit the token budget;
the runner verifies their distinct trace counters independently.

Missing labels are reported as null, not zero or perfect recall. Abstention
questions exercise the pipeline and correctness checks but receive no retrieval
coverage score: this reader-free check cannot establish answerability or
abstention quality. Source-turn/session coverage also cannot establish whether
the decisive answer span was packed. Answer accuracy requires a fresh local
open model reader and judge whenever the complete reader request changes;
frozen answers must never be scored as responses to new contexts.

## Refresh and baseline review

To reproduce the export on the existing recording host:

```sh
python export_recording.py \
  --recording /home/qrc/bench/invmem-comparison-2026-10-27-v3 \
  --output /tmp/open-model-v1.zip
```

The exporter reads the recorded dataset loader, verified source datasets,
exact vector cache, and cached tokenizer; it makes no model requests. It fails
on missing input vectors. For a future fixture, first record changed inputs
with an explicitly pinned open model in a separate recording, then export and
review its model identity, projection, selected IDs, and source hashes.

To propose a baseline for review after an intentional retrieval change:

```sh
PYTHONPATH=src python .github/scripts/check_memory_evals.py \
  --output /tmp/memory-evals \
  --record-baseline /tmp/proposed-memory-baseline.json
```

This writes a new file and refuses to overwrite one. Compare its per-case
metrics and context changes with the accepted baseline before replacing
`fixtures/baseline.json`. CI never promotes baselines automatically.

## Sources and attribution

The fixture is evaluation data, separate from the MIT-licensed library code:

- **LoCoMo**, Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal,
  Francesco Barbieri, and Yuwei Fang, *Evaluating Very Long-Term Conversational
  Memory of LLM Agents* (ACL 2024),
  [source](https://github.com/snap-research/locomo).
  The selected data and its derived embeddings retain **CC BY-NC 4.0** terms;
  see `fixtures/LICENSE-LoCoMo.txt`. Selection, serialization, role projection,
  and embedding projection are modifications for this research regression set.
- **LongMemEval**, Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang,
  and Dong Yu, *Benchmarking Chat Assistants on Long-Term Interactive Memory*
  (ICLR 2025), [source](https://github.com/xiaowu0162/LongMemEval) and
  [cleaned data](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
  Upstream MIT notice is retained in `fixtures/LICENSE-LongMemEval.txt`.
- **tiktoken**, OpenAI, [source](https://github.com/openai/tiktoken), MIT;
  its frozen vocabulary notice is in `fixtures/LICENSE-tiktoken.txt`.

Model weights are not included. The archive contains only the selected public
histories, question text, annotations, frozen vectors, provenance, and tokenizer.
