#!/usr/bin/env python3
"""Run LongMemEval benchmark: Mnemotree vs Obsidian vs Markdown baseline.

3-way comparison on long-term memory evaluation with:
- information extraction, multi-session reasoning, knowledge updates,
  temporal reasoning, abstention

Usage:
    .venv/bin/python benchmarks/run_longmemeval_comparison.py --limit-cases 20
    .venv/bin/python benchmarks/run_longmemeval_comparison.py --limit-cases 20 --no-judge
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

from benchmarks.adapters.base import CaseResult
from benchmarks.lib.store_factory import create_memory_core
from lib.markdown_baseline import MarkdownBaselineMemory
from lib.obsidian_baseline import ObsidianBaselineMemory


# --- Data loading ---

DEFAULT_DATA = WORKSPACE_ROOT / "mnemotree-adapters" / "benchmarks" / "datasets" / "longmemeval_oracle.json"


def load_longmemeval(data_path: Path, limit: int = 0, stratified: bool = False) -> list[dict]:
    """Load LongMemEval oracle dataset.

    If stratified=True, sample equally across question_type categories
    instead of taking the first N (which are all the same type).
    """
    with data_path.open() as f:
        data = json.load(f)

    if limit > 0 and stratified:
        from collections import defaultdict
        by_type: dict[str, list[dict]] = defaultdict(list)
        for item in data:
            by_type[item.get("question_type", "unknown")].append(item)

        n_types = len(by_type)
        per_type = max(1, limit // n_types)
        sampled = []
        for qtype in sorted(by_type):
            sampled.extend(by_type[qtype][:per_type])
        # Fill remaining slots from largest categories
        remaining = limit - len(sampled)
        if remaining > 0:
            for qtype in sorted(by_type, key=lambda t: len(by_type[t]), reverse=True):
                extra = by_type[qtype][per_type:per_type + remaining]
                sampled.extend(extra)
                remaining -= len(extra)
                if remaining <= 0:
                    break
        data = sampled[:limit]
    elif limit > 0:
        data = data[:limit]

    return data


def _case_sessions(item: dict) -> list[dict]:
    """Convert LongMemEval item to list of session dicts with turns."""
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])

    result = []
    for i, sess_turns in enumerate(sessions):
        date = dates[i] if i < len(dates) else None
        sid = sids[i] if i < len(sids) else f"session_{i}"
        result.append({
            "session_id": sid,
            "date": date,
            "turns": sess_turns,
        })
    return result


def _flatten_turns(sessions: list[dict]) -> list[tuple[str, str, str, str]]:
    """Flatten sessions into (content, speaker, date, session_id) tuples."""
    turns = []
    for sess in sessions:
        date = sess.get("date")
        sid = sess["session_id"]
        for turn in sess["turns"]:
            speaker = turn.get("role", "user")
            text = turn.get("content", "")
            content = f"[{date}] {speaker}: {text}" if date else f"{speaker}: {text}"
            turns.append((content, speaker, date, sid))
    return turns


# --- Runners ---

async def run_mnemotree(
    items: list[dict], *, store_dir: str, k: int,
    generate_fn: Any | None, judge_fn: Any | None,
) -> tuple[list[CaseResult], dict]:
    results: list[CaseResult] = []
    timings: list[float] = []

    for i, item in enumerate(items):
        qid = item["question_id"]
        sessions = _case_sessions(item)
        flat = _flatten_turns(sessions)

        core = await create_memory_core(
            store_dir=Path(store_dir) / "mnemotree" / qid,
            sample_id=qid,
            enable_ner=True,
            enable_keywords=False,
            reranker="none",
            retrieval_mode="hybrid",
            enable_bm25=True,
        )

        for content, speaker, date, sid in flat:
            await core.remember(
                content=content,
                context={"speaker": speaker, "date": date, "session_id": sid},
                analyze=False, summarize=False,
            )

        if (i + 1) <= 1 or (i + 1) % 5 == 0:
            print(f"    [mnemotree] Ingested {len(flat)} turns for case {i+1}/{len(items)}")

        t0 = time.perf_counter()
        retrieved = await core.recall(item["question"], limit=k, scoring=True, update_access=False)
        context = "\n".join(f"[{j+1}] {m.content}" for j, m in enumerate(retrieved))

        predicted = ""
        if generate_fn:
            predicted = await generate_fn(context, item["question"])

        score = 0.0
        if judge_fn and predicted:
            score = await judge_fn(item["question"], item["answer"], predicted)

        timings.append((time.perf_counter() - t0) * 1000)

        results.append(CaseResult(
            case_id=qid,
            question=item["question"],
            predicted=predicted,
            expected=item["answer"],
            score=score,
            category=item.get("question_type", "unknown"),
        ))

    avg_ms = sum(timings) / len(timings) if timings else 0
    return results, {"avg_case_ms": avg_ms}


async def _run_baseline(
    baseline_cls, baseline_name: str, items: list[dict], *, k: int,
    generate_fn: Any | None, judge_fn: Any | None, **kwargs,
) -> tuple[list[CaseResult], dict]:
    results: list[CaseResult] = []
    timings: list[float] = []

    for i, item in enumerate(items):
        qid = item["question_id"]
        sessions = _case_sessions(item)
        flat = _flatten_turns(sessions)

        bl = baseline_cls(**kwargs)
        for content, speaker, date, sid in flat:
            await bl.remember(
                content=content,
                context={"speaker": speaker, "date": date, "session_id": sid},
            )

        if (i + 1) <= 1 or (i + 1) % 5 == 0:
            print(f"    [{baseline_name}] Ingested {len(flat)} turns for case {i+1}/{len(items)}")

        t0 = time.perf_counter()
        retrieved = await bl.recall(item["question"], limit=k)
        context = "\n".join(f"[{j+1}] {m.content}" for j, m in enumerate(retrieved))

        predicted = ""
        if generate_fn:
            predicted = await generate_fn(context, item["question"])

        score = 0.0
        if judge_fn and predicted:
            score = await judge_fn(item["question"], item["answer"], predicted)

        timings.append((time.perf_counter() - t0) * 1000)

        results.append(CaseResult(
            case_id=qid,
            question=item["question"],
            predicted=predicted,
            expected=item["answer"],
            score=score,
            category=item.get("question_type", "unknown"),
        ))

    avg_ms = sum(timings) / len(timings) if timings else 0
    return results, {"avg_case_ms": avg_ms}


# --- Aggregation & display ---

def _aggregate(results: list[CaseResult]) -> dict:
    if not results:
        return {"accuracy": 0.0, "total": 0}
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.category or "unknown", []).append(r.score)

    def _mean(v): return sum(v) / len(v) if v else 0.0

    return {
        "accuracy": _mean([r.score for r in results]),
        "total": len(results),
        "by_category": {
            cat: {"accuracy": _mean(scores), "count": len(scores)}
            for cat, scores in sorted(by_cat.items())
        },
    }


def _print_comparison(mt: dict, ob: dict, bl: dict, mt_s: dict, ob_s: dict, bl_s: dict) -> None:
    print(f"\n{'='*80}")
    print(f"  LONGMEMEVAL COMPARISON: Mnemotree vs Obsidian vs Markdown-TF-IDF")
    print(f"{'='*80}")
    print(f"  {'Metric':<30} {'Mnemotree':>10} {'Obsidian':>10} {'Markdown':>10}  {'MT-MD':>7} {'MT-OB':>7}")
    print(f"  {'-'*79}")

    mt_a, ob_a, bl_a = mt["accuracy"], ob["accuracy"], bl["accuracy"]
    print(f"  {'Overall accuracy':<30} {mt_a:>9.1%} {ob_a:>9.1%} {bl_a:>9.1%}  {mt_a-bl_a:>+6.1%} {mt_a-ob_a:>+6.1%}")

    all_cats = sorted(set(mt.get("by_category", {})) | set(ob.get("by_category", {})) | set(bl.get("by_category", {})))
    for cat in all_cats:
        mv = mt.get("by_category", {}).get(cat, {}).get("accuracy", 0)
        ov = ob.get("by_category", {}).get(cat, {}).get("accuracy", 0)
        bv = bl.get("by_category", {}).get(cat, {}).get("accuracy", 0)
        n = mt.get("by_category", {}).get(cat, {}).get("count", 0)
        print(f"  {cat:<30} {mv:>9.1%} {ov:>9.1%} {bv:>9.1%}  {mv-bv:>+6.1%} {mv-ov:>+6.1%}  (n={n})")

    print(f"\n  {'Avg latency (ms)':<30} {mt_s.get('avg_case_ms',0):>10.1f} {ob_s.get('avg_case_ms',0):>10.1f} {bl_s.get('avg_case_ms',0):>10.1f}")


# --- Main ---

def parse_args():
    p = argparse.ArgumentParser(description="LongMemEval: 3-way comparison")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--stratified", action="store_true", help="Sample equally across question types")
    p.add_argument("--store-dir", default=str(WORKSPACE_ROOT / "artifacts" / "stores" / "longmemeval_comparison"))
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--output", default="benchmarks/results/longmemeval_comparison.json")
    return p.parse_args()


async def main():
    args = parse_args()

    print(f"Loading LongMemEval oracle...")
    items = load_longmemeval(Path(args.data), limit=args.limit_cases, stratified=args.stratified)
    print(f"  {len(items)} cases loaded")

    from collections import Counter
    types = Counter(d.get("question_type", "?") for d in items)
    print(f"  Categories: {dict(types)}")

    generate_fn = None
    judge_fn = None
    if not args.no_judge:
        from benchmarks.lib.answering import generate_answer, judge_answer, get_openai_client
        client, model = get_openai_client()
        judge_model = args.judge_model

        async def generate_fn(ctx, q):
            return await generate_answer(client, model, ctx, q)

        async def judge_fn(q, exp, pred):
            return await judge_answer(client, judge_model, q, exp, pred)

    # 1. Mnemotree
    print(f"\n[1/3] Running Mnemotree...")
    mt_results, mt_stats = await run_mnemotree(
        items, store_dir=args.store_dir, k=args.k,
        generate_fn=generate_fn, judge_fn=judge_fn,
    )
    mt_summary = _aggregate(mt_results)
    print(f"  Mnemotree accuracy: {mt_summary['accuracy']:.1%}")

    # 2. Obsidian
    print(f"\n[2/3] Running Obsidian wiki-link baseline...")
    ob_results, ob_stats = await _run_baseline(
        ObsidianBaselineMemory, "obsidian", items, k=args.k,
        generate_fn=generate_fn, judge_fn=judge_fn,
        graph_boost=0.3, backlink_depth=1,
    )
    ob_summary = _aggregate(ob_results)
    print(f"  Obsidian accuracy: {ob_summary['accuracy']:.1%}")

    # 3. Markdown
    print(f"\n[3/3] Running Markdown-TF-IDF baseline...")
    bl_results, bl_stats = await _run_baseline(
        MarkdownBaselineMemory, "markdown", items, k=args.k,
        generate_fn=generate_fn, judge_fn=judge_fn,
        mode="tfidf",
    )
    bl_summary = _aggregate(bl_results)
    print(f"  Markdown accuracy: {bl_summary['accuracy']:.1%}")

    _print_comparison(mt_summary, ob_summary, bl_summary, mt_stats, ob_stats, bl_stats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "longmemeval",
        "split": "oracle",
        "num_cases": len(items),
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
