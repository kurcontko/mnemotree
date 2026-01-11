# Benchmarks

This directory contains a configurable evaluation harness for retrieval quality and
optional answer-level judging. The legacy entrypoints now call the new harness.

## Quick start

Run retrieval-only metrics on an in-memory Chroma store:

```bash
python benchmarks/evaluate.py --store chroma --mode lite
```

Run Chroma with a persistent SQLite graph index (entity graph) enabled:

```bash
python benchmarks/evaluate.py --store chroma-graph --mode lite
```

Run against a local SQLite + sqlite-vec store:

```bash
python benchmarks/evaluate.py --store sqlite-vec --mode lite
```

Include answer-level metrics (requires an LLM and API credentials):

```bash
python benchmarks/evaluate.py --answer-eval --answer-model gpt-4.1-mini --judge-model gpt-4.1-mini
```

Use Neo4j (set credentials via env vars):

```bash
export MNEMOTREE_NEO4J_URI=bolt://localhost:7687
export MNEMOTREE_NEO4J_USER=neo4j
export MNEMOTREE_NEO4J_PASSWORD=password
python benchmarks/evaluate.py --store neo4j --mode pro
```

## What gets measured

- Retrieval effectiveness: precision@k, recall@k, MRR, and NDCG@k.
- Optional answer-level judging: answer relevance and faithfulness (LLM-judged),
  plus context precision/recall computed from expected contexts.

## Dataset schema

`benchmarks/data/memories.jsonl` fields:

- `content` (required)
- `memory_type` (optional)
- `concepts` (optional list of tags)
- `entities` (optional dict)
- `emotion` (optional)
- `importance` (optional float)

`benchmarks/data/test_queries.jsonl` fields:

- `query` (required)
- `expected_memories` (optional list of full memory contents)
- `expected_ids` (optional list of memory_ids if you seed deterministic ids)
- `expected_answer` (optional, for downstream use)
- `description` (optional)
- `query_type` (optional)

## Notes

- `--mode lite` uses local embeddings and requires `sentence-transformers`.
- `--store sqlite-vec` requires the `sqlite-vec` extension to be installed and loadable.
- `--enable-ner` or `--enable-keywords` requires the spaCy model `en_core_web_sm`.
- The legacy scripts `benchmarks/eval_memory_full.py` and
  `benchmarks/eval_memory_baseline.py` call the new harness with defaults.

### Preset: chromadb + sqlite graph + ner + rrf

```bash
python benchmarks/evaluate.py --store chroma-graph --enable-ner --retrieval-mode rrf --mode lite \
  --output benchmarks/results/evaluation_chromadb_sqlite_graph_ner_rrf.json
```

Or use the convenience preset:

```bash
python benchmarks/eval_chroma_sqlite_graph_ner_rrf.py
```

Enable BM25 as an additional RRF signal:

```bash
python benchmarks/evaluate.py --store chroma-graph --enable-ner --retrieval-mode rrf --enable-bm25 --mode lite \
  --output benchmarks/results/evaluation_chromadb_sqlite_graph_ner_rrf_bm25.json
```

Or use the convenience preset:

```bash
python benchmarks/eval_chroma_sqlite_graph_ner_rrf_bm25.py
```
