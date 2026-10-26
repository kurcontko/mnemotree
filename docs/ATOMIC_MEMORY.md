# Atomic-memory experiment

Branch `experiment/atomic-memory` is stacked on `feature/evidence-core` at
`f4e36be`. The parent remains the raw-evidence baseline. Use a fresh database for
this experiment; do not share an experiment database with the parent branch.

## What is borrowed

The conceptual starting points are Tencent's standalone L1 statements, links to
source messages, and attribution rules. We inspected TencentDB-Agent-Memory at
commit `6a4bb55ea54834d8265ce52024d3e3c75db30d38`:

- [L1 prompt and background-message formatting](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/6a4bb55ea54834d8265ce52024d3e3c75db30d38/MemoryCore/src/core/prompts/l1-extraction.ts)
- [L1 extraction pipeline](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/6a4bb55ea54834d8265ce52024d3e3c75db30d38/MemoryCore/src/core/record/l1-extractor.ts)

The Python implementation and prompt are independently written. No Tencent code
or prompt text is copied. We omit scene generation, importance thresholds,
persona categories, deduplication decisions, automatic merging, and profiles.

## Ingestion

An optional `Extractor` produces `AtomDraft(text, citations)` from ordered
passages. Each citation names a passage ID and an exact, unambiguous source quote.
The engine resolves quotes to original character offsets and rejects missing,
ambiguous, blank, or out-of-batch citations. Quote validity proves provenance,
not that a model's statement logically follows from its quotes. Atom text is
used for retrieval; answer context contains original source text.

Calls are bounded by `extraction_batch_chars` of source text (default 12,000),
plus JSON metadata and the prompt. This is a character cap, not a model-context
token guarantee. Long messages are split losslessly; extraction cannot resolve
references across batch boundaries. At most 64 atoms per call are accepted.
Repeated occurrences stay separate, and conflicting statements are retained.

The database pins provider/model/revision/prompt hash, batching configuration,
and pipeline version. Changing these requires a fresh database. A provider's
revision label alone does not pin remote weights. Reopening without an extractor
permits reads but rejects new writes to an extracted corpus. Replays of already
acknowledged requests return their receipt without repeating extraction.

Source records, raw indexes, atoms, citation spans, atom indexes, and the receipt
commit in one transaction. Extraction/embedding failures abort the add. Deleting
any cited source invalidates the whole dependent atom and its indexes. Remaining
independent raw sources are preserved. An empty extraction result is valid and
still persists the raw messages. The extra inference latency is part of Add.

## Retrieval and ablations

`SearchOptions.representation` accepts `raw` (default), `atoms`, or `combined`.
`atoms` searches only the derived indexes, then retrieves original chunks
overlapping their citations. `combined` adds these candidates to the raw channels
before RRF. Duplicate chunk hits contribute once per channel; chunks expanded
from an atom share its rank. `atom_weight` scales atom-channel contributions.

`candidate_limit` caps each raw channel and each atom channel before citation
expansion. Expansion can produce more source chunks. Returned evidence count
and exact token budget remain bounded. The trace includes atom text, source
spans, extractor signature, channel ranks, and scores for debugging. It may
contain sensitive source material; it is not included in AML HTTP responses.

Install `.[embeddings]` for the optional OpenAI-compatible extraction transport.
Configure `examples/evidence/atomic.json` with a permitted model/provider, then:

```bash
export EXTRACTION_API_KEY=your-key
mnemotree-evidence --db atomic-experiment.sqlite \
  --config examples/evidence/atomic.json \
  ablate examples/evidence/corpus.jsonl examples/evidence/queries.jsonl \
  --output atomic-ablation.json
```

The ablation ingests once and evaluates raw, atoms, and combined against exactly
the same extracted corpus, tokenizer, and budget. Reports include shared Add
latency, atom count, extractor configuration, and token usage when the provider
reports it. Unreported usage is explicit. For the ingestion-cost control, also
run `evaluate` with extraction disabled on a separate fresh database. Retrieval
source coverage is not answer accuracy. The committed corpus is a tiny plumbing
example, not evidence of a quality gain.

Python callers can use `Memory(..., extractor=OpenAIExtractor(...))` or implement
the protocol in `mnemotree.evidence.atoms`. The AML adapter requires no changes:
it uses the same atomic ingestion and returns only original evidence content.
Both extraction and embeddings are opt-in; default imports do not load HTTP or
model frameworks. No paid model calls were made during implementation.
