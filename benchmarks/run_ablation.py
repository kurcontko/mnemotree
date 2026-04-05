#!/usr/bin/env python3
"""Ablation study: which mnemotree component matters most?

Runs LoCoMo with different configs to isolate the contribution of each component:
1. Full (BM25 + embeddings + NER) — the default
2. No BM25 (embeddings + NER only)
3. No NER (BM25 + embeddings, no entity graph)
4. BM25-only (no embeddings, no NER)
5. Embeddings-only (no BM25, no NER)

Usage:
    OPENAI_BASE_URL=http://localhost:30000/v1 OPENAI_MODEL=nemotron \
    .venv/bin/python benchmarks/run_ablation.py --limit-cases 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
WORKTREE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = WORKTREE_ROOT.parent

if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_src = os.environ.get("MNEMOTREE_SRC", str(WORKTREE_ROOT / "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from dotenv import load_dotenv
    load_dotenv(WORKSPACE_ROOT / "EasyLocomo" / ".env")
except ImportError:
    pass

from benchmarks.adapters.locomo import LoCoMoAdapter
from benchmarks.adapters.base import BenchmarkCase, CaseResult
from benchmarks.lib.store_factory import create_memory_core


CONFIGS = {
    "full": {"enable_ner": True, "enable_bm25": True, "retrieval_mode": "hybrid", "reranker": "none"},
    "no_bm25": {"enable_ner": True, "enable_bm25": False, "retrieval_mode": "hybrid", "reranker": "none"},
    "no_ner": {"enable_ner": False, "enable_bm25": True, "retrieval_mode": "hybrid", "reranker": "none"},
    "bm25_only": {"enable_ner": False, "enable_bm25": True, "retrieval_mode": "basic", "reranker": "none"},
    "embed_only": {"enable_ner": False, "enable_bm25": False, "retrieval_mode": "basic", "reranker": "none"},
}


async def run_config(
    name: str, config: dict, adapter: LoCoMoAdapter, cases: list[BenchmarkCase],
    *, store_dir: str, k: int, generate_fn, judge_fn,
) -> tuple[str, dict]:
    results: list[CaseResult] = []
    ingested: set[str] = set()
    cores: dict[str, Any] = {}

    for i, case in enumerate(cases):
        sid = case.metadata.get("sample_id", case.case_id)

        if sid not in cores:
            core = await create_memory_core(
                store_dir=Path(store_dir) / name / sid,
                sample_id=sid,
                enable_keywords=False,
                **config,
            )
            cores[sid] = core

        if sid not in ingested:
            n = await adapter.ingest(cores[sid], case)
            ingested.add(sid)
            print(f"    [{name}] Ingested {n} memories for {sid}")

        result = await adapter.run_case(
            cores[sid], case, k=k,
            generate_answer=generate_fn, judge_answer=judge_fn,
        )
        results.append(result)

    if not results:
        return name, {"accuracy": 0.0, "total": 0}

    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.category or "unknown", []).append(r.score)

    def _m(v): return sum(v) / len(v) if v else 0.0

    summary = {
        "accuracy": _m([r.score for r in results]),
        "total": len(results),
        "by_category": {
            cat: {"accuracy": _m(scores), "count": len(scores)}
            for cat, scores in sorted(by_cat.items())
        },
    }
    return name, summary


async def main():
    p = argparse.ArgumentParser(description="Ablation study on LoCoMo")
    p.add_argument("--limit-cases", type=int, default=5)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--store-dir", default=str(WORKSPACE_ROOT / "artifacts" / "stores" / "ablation"))
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--output", default="benchmarks/results/ablation.json")
    args = p.parse_args()

    adapter = LoCoMoAdapter()
    dataset = adapter.load(split="locomo10")
    cases = dataset.cases[:args.limit_cases]
    print(f"Loaded {len(cases)} LoCoMo cases")

    from benchmarks.lib.answering import generate_answer, judge_answer, get_openai_client
    client, model = get_openai_client()
    judge_model = args.judge_model

    async def gen_fn(ctx, q):
        return await generate_answer(client, model, ctx, q)

    async def jdg_fn(q, exp, pred):
        return await judge_answer(client, judge_model, q, exp, pred)

    results = {}
    for config_name, config in CONFIGS.items():
        print(f"\n=== Running {config_name} ===")
        name, summary = await run_config(
            config_name, config, adapter, cases,
            store_dir=args.store_dir, k=args.k,
            generate_fn=gen_fn, judge_fn=jdg_fn,
        )
        results[name] = summary
        print(f"  {name}: {summary['accuracy']:.1%}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY: LoCoMo ({len(cases)} cases)")
    print(f"{'='*60}")
    print(f"  {'Config':<20} {'Accuracy':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    full_acc = results.get("full", {}).get("accuracy", 0)
    for name in CONFIGS:
        acc = results.get(name, {}).get("accuracy", 0)
        delta = acc - full_acc
        marker = " (baseline)" if name == "full" else ""
        print(f"  {name:<20} {acc:>9.1%} {delta:>+9.1%}{marker}")

    # Per-category for full config
    full_cats = results.get("full", {}).get("by_category", {})
    if full_cats:
        print(f"\n  Full config by category:")
        for cat, v in sorted(full_cats.items()):
            print(f"    {cat}: {v['accuracy']:.1%} (n={v['count']})")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"benchmark": "ablation_locomo", "configs": results, "num_cases": len(cases)}, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
