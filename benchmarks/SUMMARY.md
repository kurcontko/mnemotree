# Mnemotree Benchmark Summary

## LoCoMo Benchmark (External — Long-term Conversational Memory)

10 conversations, 1,986 questions across 5 categories. Judge: gpt-4o-mini.

| Version | Config | Strict | Partial Credit | vs Mem0 |
|---|---|---|---|---|
| v4 | gpt-5.2, hybrid+cross-encoder | 41.9% | 53.9% | -13.0 |
| v11 | + BGE embeddings, k=20 | 52.3% | 62.3% | -4.6 |
| Mem0 | published (excl. adversarial) | — | 66.9% | — |
| **v16** | **ENGRAM + full pipeline** | **61.3%** | **70.0%** | **+3.1** |

### v16 Per-Category Breakdown

| Category | Questions | Strict | Partial Credit |
|---|---|---|---|
| 1 — Single-hop | 282 | 37.2% | 60.1% |
| 2 — Temporal | 321 | 56.7% | 67.0% |
| 3 — Multi-hop | 96 | 34.4% | 42.7% |
| 4 — Open-domain | 841 | 67.2% | 75.1% |
| 5 — Adversarial | 446 | 74.4% | 74.4% |

### v16 Configuration

- LLM: openai/gpt-5.2 (via OpenRouter)
- Embeddings: BAAI/bge-base-en-v1.5 on CUDA
- Reranker: cross-encoder/ms-marco-MiniLM-L-12-v2 (top 100 candidates)
- NER: GLiNER + keywords + PRF
- k=20 per category (25 for multi-hop)
- ENGRAM fact-as-memory architecture
- Local router (ModernBERT), no extractor
- Per-sample store isolation

## Internal Dataset Baselines

Evaluated on 80 memories, 78 queries (culinary + general knowledge).

| Configuration | Precision@5 | Recall@5 | MRR | NDCG@5 | Semantic Sim |
|---|---|---|---|---|---|
| Baseline (no NER/scoring) | 0.241 | 0.740 | 0.694 | 0.787 | 0.570 |
| Full (NER + scoring) | 0.408 | 0.846 | 0.833 | 0.850 | — |
| **Delta** | **+69.2%** | **+14.3%** | **+20.1%** | **+8.0%** | — |

Backend: Neo4j. LLM answer relevance (full config): 0.789.

## Retrieval Pipeline Features

Features available for benchmarking (enabled via CLI flags):

| Feature | Flag | Status |
|---|---|---|
| Hybrid RRF retrieval | `--retrieval-mode rrf` | Ready |
| BM25 sparse signal | `--enable-bm25` | Ready |
| NER entity extraction | `--enable-ner` | Ready (spaCy/GLiNER/LLM/DistilBERT) |
| Keyword extraction | `--enable-keywords` | Ready |
| FlashRank reranking | `--enable-reranker` | Ready |
| PRF query expansion | `--enable-prf` | Ready |
| Intent-aware filtering | via code | Ready (SimpleMem) |
| HyDE embedding | via code | Ready |
| MS-RAG decomposition | via code | Ready |
| MAGMA graph traversal | via code | Ready |
| Conflict detection | via code | Ready (ConflictJudge) |

## Improvement Targets

### LoCoMo (current best: v16 — 70.0% partial credit)
- Partial credit: >=75% (currently 70.0%)
- Multi-hop category: >=55% (currently 42.7% — weakest)
- Single-hop category: >=70% (currently 60.1%)

### Internal dataset
- Precision@5: >=0.50 (currently 0.408 with NER)
- Recall@5: >=0.90 (currently 0.846)

### Future benchmarks
- LongMemEval: >=80% accuracy (temporal reasoning, multi-hop, contradiction)
- MemoryArena: Multi-agent memory benchmark
- AMA-Bench: Active memory agent benchmark (conflict resolution focus)

## Architecture Improvements Since Baseline

Features implemented that should improve benchmark scores on re-evaluation:

1. **STITCH contextual intent** — +35.6% retrieval accuracy in paper
2. **Bi-temporal fields** — TSM-style event_time vs storage timestamp
3. **Memory evolution** — A-MEM pattern, update related memories on store
4. **MAGMA four-graph traversal** — Semantic + temporal + causal + entity
5. **ConflictJudge** — AMA pattern fast conflict detection
6. **Observer/Reflector** — Mastra OM write-only observation pattern
7. **Deduplication at ingest** — SimpleMem Stage 2
8. **Fact decomposition** — Atomic fact extraction for compound content
9. **Namespace isolation** — User/conversation filtering at store layer
10. **Internal protocols** — LangChain decoupled, lite mode possible

## Running Benchmarks

```bash
# Quick baseline (no API keys needed)
python benchmarks/evaluate.py --store sqlite-vec --mode lite

# Full evaluation with all features
python benchmarks/evaluate.py --store chroma-graph --enable-ner --enable-keywords \
  --retrieval-mode rrf --enable-bm25 --mode lite \
  --output benchmarks/results/latest.json

# With answer evaluation (requires OpenAI API key)
python benchmarks/evaluate.py --store chroma-graph --enable-ner --retrieval-mode rrf \
  --answer-eval --answer-model gpt-4.1-mini --judge-model gpt-4.1-mini

# Ablation study (4 configurations compared)
bash benchmarks/run_ablation.sh

# Compare results
python benchmarks/compare_results.py \
  benchmarks/results/baseline.json benchmarks/results/candidate.json
```

## Files

| File | Description |
|---|---|
| `evaluate.py` | Main evaluation harness (1053 lines, 30+ CLI options) |
| `compare_results.py` | Compare baseline vs candidate results |
| `run_ablation.sh` | Automated 4-run ablation study |
| `data/memories.jsonl` | 80 test memories (culinary + facts) |
| `data/test_queries.jsonl` | 78 test queries with expected results |
| `results/` | JSON output from evaluation runs |
