#!/usr/bin/env python3
"""Run LoCoMo benchmark: Mnemotree vs Markdown vs Obsidian baselines.

Same questions, same LLM judge, different retrieval backends.
This is the real test — multi-turn conversational memory with
multi-hop, temporal, and adversarial queries.

Usage:
    .venv/bin/python benchmarks/run_locomo_comparison.py
    .venv/bin/python benchmarks/run_locomo_comparison.py --limit-cases 50
    .venv/bin/python benchmarks/run_locomo_comparison.py --no-judge  # retrieval-only
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
from lib.markdown_baseline import MarkdownBaselineMemory
from lib.obsidian_baseline import ObsidianBaselineMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoCoMo: Mnemotree vs Markdown baseline.")
    parser.add_argument("--split", default="locomo10")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=20, help="Retrieval top-k.")
    parser.add_argument("--limit-cases", type=int, default=0, help="0 = all cases.")
    parser.add_argument("--store-dir", default=str(WORKSPACE_ROOT / "artifacts" / "stores" / "locomo_comparison"))
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM generation+judging (retrieval metrics only).")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--output", default="benchmarks/results/locomo_comparison.json")
    return parser.parse_args()


async def run_mnemotree(
    adapter: LoCoMoAdapter,
    cases: list[BenchmarkCase],
    *,
    store_dir: str,
    k: int,
    generate_fn: Any | None,
    judge_fn: Any | None,
) -> tuple[list[CaseResult], dict[str, float]]:
    """Run LoCoMo through mnemotree with full pipeline."""
    results: list[CaseResult] = []
    ingested: set[str] = set()
    cores: dict[str, Any] = {}
    timings: list[float] = []

    for i, case in enumerate(cases):
        sid = case.metadata.get("sample_id", case.case_id)

        if sid not in cores:
            core = await create_memory_core(
                store_dir=Path(store_dir) / "mnemotree",
                sample_id=sid,
                enable_ner=True,
                enable_keywords=False,
                reranker="none",
                retrieval_mode="hybrid",
                enable_bm25=True,
            )
            cores[sid] = core

        if sid not in ingested:
            n = await adapter.ingest(cores[sid], case)
            ingested.add(sid)
            print(f"    [mnemotree] Ingested {n} memories for {sid}")

        t0 = time.perf_counter()
        result = await adapter.run_case(
            cores[sid], case, k=k,
            generate_answer=generate_fn, judge_answer=judge_fn,
        )
        timings.append((time.perf_counter() - t0) * 1000)
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"    [mnemotree] {i+1}/{len(cases)} done")

    avg_ms = sum(timings) / len(timings) if timings else 0
    return results, {"avg_case_ms": avg_ms}


async def run_markdown_baseline(
    adapter: LoCoMoAdapter,
    cases: list[BenchmarkCase],
    *,
    k: int,
    generate_fn: Any | None,
    judge_fn: Any | None,
) -> tuple[list[CaseResult], dict[str, float]]:
    """Run LoCoMo through markdown TF-IDF baseline."""
    results: list[CaseResult] = []
    ingested: set[str] = set()
    baselines: dict[str, MarkdownBaselineMemory] = {}
    timings: list[float] = []
    total_tokens = 0

    for i, case in enumerate(cases):
        sid = case.metadata.get("sample_id", case.case_id)

        if sid not in baselines:
            baselines[sid] = MarkdownBaselineMemory(mode="tfidf")

        if sid not in ingested:
            bl = baselines[sid]
            count = 0
            for session in case.sessions:
                for turn in session.turns:
                    content = (
                        f"[{session.date}] {turn['speaker']}: {turn['text']}"
                        if session.date
                        else f"{turn['speaker']}: {turn['text']}"
                    )
                    await bl.remember(content=content)
                    count += 1
            ingested.add(sid)
            total_tokens += bl.stats.total_tokens
            print(f"    [markdown]  Ingested {count} memories for {sid} ({bl.stats.total_tokens} tokens)")

        bl = baselines[sid]
        t0 = time.perf_counter()

        # Retrieval
        retrieved = await bl.recall(case.question, limit=k)
        retrieved_ids = [r.memory_id for r in retrieved]
        context = "\n".join(f"[{j+1}] {m.content}" for j, m in enumerate(retrieved))

        predicted = ""
        if generate_fn:
            predicted = await generate_fn(context, case.question)

        score = 0.0
        if judge_fn and predicted:
            score = await judge_fn(case.question, case.expected.text, predicted)

        timings.append((time.perf_counter() - t0) * 1000)

        results.append(CaseResult(
            case_id=case.case_id,
            question=case.question,
            predicted=predicted,
            expected=case.expected.text,
            score=score,
            category=case.category,
            retrieved_ids=retrieved_ids,
        ))

        if (i + 1) % 50 == 0:
            print(f"    [markdown]  {i+1}/{len(cases)} done")

    avg_ms = sum(timings) / len(timings) if timings else 0
    return results, {"avg_case_ms": avg_ms, "total_tokens_stored": total_tokens}


async def run_obsidian_baseline(
    adapter: LoCoMoAdapter,
    cases: list[BenchmarkCase],
    *,
    k: int,
    generate_fn: Any | None,
    judge_fn: Any | None,
) -> tuple[list[CaseResult], dict[str, float]]:
    """Run LoCoMo through Obsidian-style wiki-link + graph baseline."""
    results: list[CaseResult] = []
    ingested: set[str] = set()
    vaults: dict[str, ObsidianBaselineMemory] = {}
    timings: list[float] = []
    total_tokens = 0

    for i, case in enumerate(cases):
        sid = case.metadata.get("sample_id", case.case_id)

        if sid not in vaults:
            vaults[sid] = ObsidianBaselineMemory(graph_boost=0.3, backlink_depth=1)

        if sid not in ingested:
            vault = vaults[sid]
            count = 0
            for session in case.sessions:
                for turn in session.turns:
                    content = (
                        f"[{session.date}] {turn['speaker']}: {turn['text']}"
                        if session.date
                        else f"{turn['speaker']}: {turn['text']}"
                    )
                    await vault.remember(
                        content=content,
                        context={
                            "speaker": turn["speaker"],
                            "date": session.date,
                            "session_id": session.session_id,
                        },
                    )
                    count += 1
            ingested.add(sid)
            total_tokens += vault.stats.total_tokens
            entities = vault.stats.total_entities
            links = vault.stats.total_links
            print(f"    [obsidian]  Ingested {count} notes for {sid} ({total_tokens} tokens, {entities} entities, {links} links)")

        vault = vaults[sid]
        t0 = time.perf_counter()

        retrieved = await vault.recall(case.question, limit=k)
        retrieved_ids = [r.memory_id for r in retrieved]
        context = "\n".join(f"[{j+1}] {m.content}" for j, m in enumerate(retrieved))

        predicted = ""
        if generate_fn:
            predicted = await generate_fn(context, case.question)

        score = 0.0
        if judge_fn and predicted:
            score = await judge_fn(case.question, case.expected.text, predicted)

        timings.append((time.perf_counter() - t0) * 1000)

        results.append(CaseResult(
            case_id=case.case_id,
            question=case.question,
            predicted=predicted,
            expected=case.expected.text,
            score=score,
            category=case.category,
            retrieved_ids=retrieved_ids,
        ))

        if (i + 1) % 50 == 0:
            print(f"    [obsidian]  {i+1}/{len(cases)} done")

    avg_ms = sum(timings) / len(timings) if timings else 0
    return results, {"avg_case_ms": avg_ms, "total_tokens_stored": total_tokens}


def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    """Compute accuracy by category."""
    if not results:
        return {"accuracy": 0.0, "total": 0}

    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.category or "unknown", []).append(r.score)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "accuracy": _mean([r.score for r in results]),
        "total": len(results),
        "by_category": {
            cat: {"accuracy": _mean(scores), "count": len(scores)}
            for cat, scores in sorted(by_cat.items())
        },
    }


def _print_comparison(
    mt_summary: dict, bl_summary: dict, ob_summary: dict,
    mt_stats: dict, bl_stats: dict, ob_stats: dict,
) -> None:
    print(f"\n{'='*80}")
    print(f"  LOCOMO COMPARISON: Mnemotree vs Obsidian vs Markdown-TF-IDF")
    print(f"{'='*80}")
    print(f"  {'Metric':<25} {'Mnemotree':>11} {'Obsidian':>11} {'Markdown':>11}  {'MT-MD':>8} {'MT-OB':>8}")
    print(f"  {'-'*79}")

    mt_acc = mt_summary.get("accuracy", 0)
    bl_acc = bl_summary.get("accuracy", 0)
    ob_acc = ob_summary.get("accuracy", 0)
    d_md = mt_acc - bl_acc
    d_ob = mt_acc - ob_acc
    print(f"  {'Overall accuracy':<25} {mt_acc:>10.1%} {ob_acc:>10.1%} {bl_acc:>10.1%}  {d_md:>+7.1%} {d_ob:>+7.1%}")

    # Per-category
    mt_cats = mt_summary.get("by_category", {})
    bl_cats = bl_summary.get("by_category", {})
    ob_cats = ob_summary.get("by_category", {})
    all_cats = sorted(set(mt_cats) | set(bl_cats) | set(ob_cats))
    for cat in all_cats:
        mt_v = mt_cats.get(cat, {}).get("accuracy", 0)
        bl_v = bl_cats.get(cat, {}).get("accuracy", 0)
        ob_v = ob_cats.get(cat, {}).get("accuracy", 0)
        count = mt_cats.get(cat, {}).get("count", 0)
        d1 = mt_v - bl_v
        d2 = mt_v - ob_v
        print(f"  {cat:<25} {mt_v:>10.1%} {ob_v:>10.1%} {bl_v:>10.1%}  {d1:>+7.1%} {d2:>+7.1%}  (n={count})")

    print(f"\n  {'Avg latency (ms)':<25} {mt_stats.get('avg_case_ms',0):>11.1f} {ob_stats.get('avg_case_ms',0):>11.1f} {bl_stats.get('avg_case_ms',0):>11.1f}")
    print(f"  MT-MD = mnemotree vs markdown delta, MT-OB = mnemotree vs obsidian delta")


async def main() -> None:
    args = parse_args()
    adapter = LoCoMoAdapter()

    print(f"Loading LoCoMo/{args.split}...")
    dataset = adapter.load(split=args.split, data_dir=args.data_dir)
    cases = dataset.cases
    if args.limit_cases > 0:
        cases = cases[:args.limit_cases]
    print(f"  {len(cases)} cases loaded")

    generate_fn = None
    judge_fn = None
    if not args.no_judge:
        from benchmarks.lib.answering import generate_answer, judge_answer, get_openai_client
        client, model = get_openai_client()
        judge_model = args.judge_model

        async def generate_fn(ctx: str, q: str) -> str:
            return await generate_answer(client, model, ctx, q)

        async def judge_fn(q: str, exp: str, pred: str) -> float:
            return await judge_answer(client, judge_model, q, exp, pred)

    # Run mnemotree
    print(f"\n[1/3] Running Mnemotree...")
    mt_results, mt_stats = await run_mnemotree(
        adapter, cases, store_dir=args.store_dir, k=args.k,
        generate_fn=generate_fn, judge_fn=judge_fn,
    )
    mt_summary = _aggregate(mt_results)
    print(f"  Mnemotree accuracy: {mt_summary['accuracy']:.1%}")

    # Run obsidian baseline
    print(f"\n[2/3] Running Obsidian wiki-link baseline...")
    ob_results, ob_stats = await run_obsidian_baseline(
        adapter, cases, k=args.k,
        generate_fn=generate_fn, judge_fn=judge_fn,
    )
    ob_summary = _aggregate(ob_results)
    print(f"  Obsidian accuracy: {ob_summary['accuracy']:.1%}")

    # Run markdown baseline
    print(f"\n[3/3] Running Markdown-TF-IDF baseline...")
    bl_results, bl_stats = await run_markdown_baseline(
        adapter, cases, k=args.k,
        generate_fn=generate_fn, judge_fn=judge_fn,
    )
    bl_summary = _aggregate(bl_results)
    print(f"  Markdown accuracy: {bl_summary['accuracy']:.1%}")

    _print_comparison(mt_summary, bl_summary, ob_summary, mt_stats, bl_stats, ob_stats)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "locomo",
        "split": args.split,
        "num_cases": len(cases),
        "k": args.k,
        "judge_enabled": not args.no_judge,
        "mnemotree": {"summary": mt_summary, "stats": mt_stats},
        "obsidian_baseline": {"summary": ob_summary, "stats": ob_stats},
        "markdown_baseline": {"summary": bl_summary, "stats": bl_stats},
    }
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
