from __future__ import annotations

import asyncio
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.evaluate import EvalConfig, run_benchmark


def main() -> None:
    """Legacy entrypoint. Prefer benchmarks/evaluate.py for configurable runs."""
    config = EvalConfig(
        data_dir=Path("benchmarks/data"),
        memories_file="memories.jsonl",
        queries_file="test_queries.jsonl",
        k_values=[1, 3, 5, 10],
        store_type="neo4j",
        mode="pro",
        scoring=True,
        enable_ner=True,
        ner_type="spacy",
        enable_keywords=False,
        retrieval_mode="basic",
        enable_bm25=False,
        rrf_k=60,
        enable_prf=False,
        prf_docs=5,
        prf_terms=8,
        enable_rrf_signal_rerank=False,
        reranker_backend="none",
        reranker_model="ms-marco-TinyBERT-L-2-v2",
        rerank_candidates=50,
        answer_eval=False,
        answer_k=5,
        answer_model=None,
        judge_model=None,
        output_path=Path("benchmarks/results/evaluation_results-full_neo4j.json"),
        dummy_embeddings=False,
    )

    results = asyncio.run(run_benchmark(config))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w") as handle:
        import json

        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
