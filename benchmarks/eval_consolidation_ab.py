#!/usr/bin/env python3
"""A/B comparison: retrieval quality with and without consolidation.

Runs the internal benchmark dataset twice:
  A) Baseline — ingest + recall (no consolidation)
  B) With consolidation — ingest, consolidate, then recall

Reports retrieval metrics side-by-side.

Usage:
    python benchmarks/eval_consolidation_ab.py [--store chroma|sqlite-vec] [--k 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mnemotree.core.memory import (
    IngestionConfig,
    MemoryCore,
    ModeDefaultsConfig,
    RecallFilters,
    RetrievalConfig,
)
from mnemotree.core.models import MemoryType
from mnemotree.core.events import EventType, MemoryEvent


def load_memories(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_queries(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


async def build_store(store_type: str, run_id: str):
    if store_type == "sqlite-vec":
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore
        store = SQLiteVecMemoryStore(db_path=":memory:")
        await store.initialize()
        return store
    else:
        from mnemotree.store.chromadb_store import ChromaMemoryStore
        store = ChromaMemoryStore(collection_name=f"ab_test_{run_id}")
        return store


async def seed_and_recall(
    store_type: str,
    memories: list[dict],
    queries: list[dict],
    k: int,
    run_id: str,
    do_consolidation: bool = False,
) -> dict[str, Any]:
    """Seed memories, optionally consolidate, then evaluate recall."""

    store = await build_store(store_type, run_id)

    # Track events
    events: list[str] = []

    memory_core = MemoryCore(
        store=store,
        mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=True),
        retrieval_config=RetrievalConfig(
            retrieval_mode="hybrid",
            enable_bm25=True,
        ),
        ingestion_config=IngestionConfig(
            auto_link_enabled=False,  # disable for fair comparison
            conflict_detection_enabled=False,
            dedup_enabled=False,
        ),
    )

    memory_core.on_event(lambda e: events.append(e.event_type.value))

    # Ingest
    t0 = time.time()
    for rec in memories:
        mt_str = rec.get("memory_type", "semantic")
        try:
            mt = MemoryType(mt_str)
        except ValueError:
            mt = MemoryType.SEMANTIC

        await memory_core.remember(
            content=rec["content"],
            memory_type=mt,
            importance=rec.get("importance", 0.5),
            tags=rec.get("tags") or rec.get("concepts", []),
        )
    ingest_time = time.time() - t0

    consolidation_stats = None
    if do_consolidation:
        # Need an LLM for consolidation — try to use a lightweight approach
        # If no LLM, we'll use the clustering-only path
        try:
            t1 = time.time()
            result = await memory_core.consolidate()
            consolidation_stats = {
                "clusters_formed": result.clusters_formed,
                "semantic_created": result.semantic_memories_created,
                "deprecated": result.memories_deprecated,
                "duration": time.time() - t1,
            }
        except RuntimeError as e:
            # No LLM configured — expected
            consolidation_stats = {"error": str(e)}

    # Evaluate recall
    hits_at_k = 0
    reciprocal_ranks = []
    total = 0

    t2 = time.time()
    for q in queries:
        query_text = q["query"]
        expected = set(q.get("expected_memories", []))
        if not expected:
            continue
        total += 1

        results = await memory_core.recall(query_text, limit=k, scoring=True)
        retrieved_contents = [r.content for r in results]

        # Check hits
        found = False
        first_rank = None
        for i, content in enumerate(retrieved_contents):
            if content in expected:
                if not found:
                    first_rank = i + 1
                    found = True

        if found:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_time = time.time() - t2

    hit_rate = hits_at_k / total if total else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        "label": "with_consolidation" if do_consolidation else "baseline",
        "num_memories": len(memories),
        "num_queries": total,
        "k": k,
        "hit_rate@k": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "ingest_time": round(ingest_time, 2),
        "recall_time": round(recall_time, 2),
        "events": dict(
            remember=events.count("remember"),
            recall=events.count("recall"),
            consolidate=events.count("consolidate"),
        ),
        "consolidation": consolidation_stats,
    }


async def run_ab(store_type: str, k: int):
    data_dir = PROJECT_ROOT / "benchmarks" / "data"
    memories = load_memories(data_dir / "memories.jsonl")
    queries = load_queries(data_dir / "test_queries.jsonl")

    print(f"Loaded {len(memories)} memories, {len(queries)} queries")
    print(f"Store: {store_type}, k={k}")
    print()

    # Run A: baseline
    print("=" * 60)
    print("RUN A: Baseline (no consolidation)")
    print("=" * 60)
    result_a = await seed_and_recall(store_type, memories, queries, k, "baseline", do_consolidation=False)
    print(json.dumps(result_a, indent=2))
    print()

    # Run B: with consolidation
    print("=" * 60)
    print("RUN B: With consolidation")
    print("=" * 60)
    result_b = await seed_and_recall(store_type, memories, queries, k, "consolidated", do_consolidation=True)
    print(json.dumps(result_b, indent=2))
    print()

    # Comparison
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    delta_hit = result_b["hit_rate@k"] - result_a["hit_rate@k"]
    delta_mrr = result_b["mrr"] - result_a["mrr"]
    print(f"  Hit Rate@{k}: {result_a['hit_rate@k']:.4f} → {result_b['hit_rate@k']:.4f} (Δ {delta_hit:+.4f})")
    print(f"  MRR:          {result_a['mrr']:.4f} → {result_b['mrr']:.4f} (Δ {delta_mrr:+.4f})")
    print(f"  Events A:     {result_a['events']}")
    print(f"  Events B:     {result_b['events']}")

    return {"baseline": result_a, "with_consolidation": result_b}


def main():
    parser = argparse.ArgumentParser(description="A/B test: baseline vs consolidation")
    parser.add_argument("--store", default="sqlite-vec", choices=["chroma", "sqlite-vec"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = asyncio.run(run_ab(args.store, args.k))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
