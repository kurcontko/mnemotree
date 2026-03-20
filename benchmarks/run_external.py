#!/usr/bin/env python3
"""Unified runner for all external memory benchmarks.

Usage:
    python benchmarks/run_external.py --benchmark locomo --split locomo10 --output-dir ../artifacts/outputs/locomo
    python benchmarks/run_external.py --benchmark longmemeval --split oracle --output-dir ../artifacts/outputs/longmemeval
    python benchmarks/run_external.py --benchmark personamem --split 32k --output-dir ../artifacts/outputs/personamem
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure benchmarks package and mnemotree are importable
SCRIPT_DIR = Path(__file__).resolve().parent
WORKTREE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = WORKTREE_ROOT.parent
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))
MNEMOTREE_ROOT = Path(os.environ.get("MNEMOTREE_ROOT", str(WORKSPACE_ROOT / "mnemotree")))
_src = os.environ.get("MNEMOTREE_SRC", str(MNEMOTREE_ROOT / "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from dotenv import load_dotenv
    load_dotenv(WORKSPACE_ROOT / "EasyLocomo" / ".env")
except ImportError:
    pass

from benchmarks.adapters.base import BenchmarkAdapter
from benchmarks.lib.answering import generate_answer, judge_answer, get_openai_client
from benchmarks.lib.results import build_result, save_result
from benchmarks.lib.store_factory import create_memory_core

ADAPTERS: dict[str, type] = {}


def _register_adapters() -> None:
    from benchmarks.adapters.locomo import LoCoMoAdapter
    from benchmarks.adapters.longmemeval import LongMemEvalAdapter
    from benchmarks.adapters.personamem import PersonaMemAdapter
    from benchmarks.adapters.prefeval import PrefEvalAdapter
    from benchmarks.adapters.halumem import HaluMemAdapter
    from benchmarks.adapters.memoryagentbench import MemoryAgentBenchAdapter

    for cls in [LoCoMoAdapter, LongMemEvalAdapter, PersonaMemAdapter, PrefEvalAdapter, HaluMemAdapter, MemoryAgentBenchAdapter]:
        ADAPTERS[cls.name] = cls


def parse_args() -> argparse.Namespace:
    _register_adapters()
    parser = argparse.ArgumentParser(description="Run an external memory benchmark against mnemotree.")
    parser.add_argument("--benchmark", required=True, choices=list(ADAPTERS.keys()), help="Benchmark to run.")
    parser.add_argument("--split", default=None, help="Dataset split (benchmark-specific).")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory.")
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "artifacts" / "outputs" / "external"), help="Output directory.")
    parser.add_argument("--k", type=int, default=20, help="Retrieval k-value.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Limit number of cases (0 = all).")
    parser.add_argument("--store-dir", default=str(WORKSPACE_ROOT / "artifacts" / "stores"), help="Store directory.")
    parser.add_argument("--reranker", default="cross_encoder", choices=["none", "flashrank", "cross_encoder"])
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini", help="LLM judge model.")
    parser.add_argument("--no-ner", action="store_true")
    parser.add_argument("--no-keywords", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    adapter_cls = ADAPTERS[args.benchmark]
    adapter = adapter_cls()

    split = args.split or "default"
    print(f"Loading {args.benchmark}/{split}...")
    dataset = adapter.load(split=split, data_dir=args.data_dir)

    cases = dataset.cases
    if args.limit_cases > 0:
        cases = cases[:args.limit_cases]
    print(f"  {len(cases)} cases loaded")

    client, model = get_openai_client()
    judge_model = args.judge_model

    # Group cases by sample_id to share stores where possible
    sample_ids = set()
    for case in cases:
        sid = case.metadata.get("sample_id", case.case_id)
        sample_ids.add(sid)

    results = []
    start = time.time()

    # For benchmarks where all cases share the same sessions (like LoCoMo),
    # we ingest once per sample. For others, ingest per case.
    ingested_samples: set[str] = set()
    cores: dict[str, object] = {}

    for i, case in enumerate(cases):
        sid = case.metadata.get("sample_id", case.case_id)

        if sid not in cores:
            core = await create_memory_core(
                store_dir=Path(args.store_dir) / f"{args.benchmark}_{split}",
                sample_id=sid,
                enable_ner=not args.no_ner,
                enable_keywords=not args.no_keywords,
                reranker=args.reranker,
            )
            cores[sid] = core

        if sid not in ingested_samples:
            n = await adapter.ingest(cores[sid], case)
            ingested_samples.add(sid)
            print(f"  Ingested {n} memories for {sid}")

        async def _generate(ctx: str, q: str) -> str:
            return await generate_answer(client, model, ctx, q)

        async def _judge(q: str, exp: str, pred: str) -> float:
            return await judge_answer(client, judge_model, q, exp, pred)

        result = await adapter.run_case(
            cores[sid], case, k=args.k,
            generate_answer=_generate, judge_answer=_judge,
        )
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(cases)} cases done ({elapsed:.0f}s)")

    elapsed = time.time() - start
    summary = adapter.aggregate(results)
    summary["elapsed_seconds"] = elapsed

    output = build_result(
        benchmark=args.benchmark,
        split=split,
        summary=summary,
        per_case_results=results,
        config=vars(args),
    )

    out_path = Path(args.output_dir) / f"{args.benchmark}_{split}.json"
    save_result(output, out_path)
    print(f"\nResults saved to {out_path}")
    print(f"Accuracy: {summary.get('accuracy', 0):.1%}  |  Total: {summary.get('total', 0)}  |  Time: {elapsed:.0f}s")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
