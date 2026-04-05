#!/usr/bin/env python3
"""Run MemoryAgentBench: Mnemotree vs Obsidian vs Markdown.

Tests agent memory across competencies: Accurate Retrieval (AR),
Conflict Resolution (CR), Long-Range Understanding (LRU).

Usage:
    .venv/bin/python benchmarks/run_mab_comparison.py --splits ar,cr --limit-questions 10
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

MAB_DATA = WORKSPACE_ROOT / "mnemotree-adapters" / "benchmarks" / "datasets" / "MemoryAgentBench"

SPLIT_TO_PARQUET = {
    "ar": "Accurate_Retrieval",
    "cr": "Conflict_Resolution",
    "ttl": "Test_Time_Learning",
    "lru": "Long_Range_Understanding",
}

CHUNK_SIZE = 512


def _chunk_context(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    lines = text.split("\n")
    chunks, current, current_len = [], [], 0
    for line in lines:
        if current_len + len(line) + 1 > chunk_size and current:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def load_mab(splits: list[str], limit_docs: int = 0, limit_questions: int = 0) -> list[dict]:
    """Load MAB cases from parquet, returning flat list of {doc_id, question, answers, chunks, split}."""
    import pandas as pd
    items = []
    for split in splits:
        parquet_name = SPLIT_TO_PARQUET.get(split, split)
        path = MAB_DATA / "data" / f"{parquet_name}-00000-of-00001.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {split}")
            continue
        df = pd.read_parquet(path)
        docs = list(df.iterrows())
        if limit_docs > 0:
            docs = docs[:limit_docs]

        for row_idx, row in docs:
            context = row["context"]
            questions = list(row["questions"])
            answers = list(row["answers"])
            chunks = _chunk_context(context)
            source = (row.get("metadata") or {}).get("source", f"row_{row_idx}")
            doc_id = f"{split}_{source}_{row_idx}"

            qa_pairs = list(zip(questions, answers))
            if limit_questions > 0:
                qa_pairs = qa_pairs[:limit_questions]

            for qi, (q, a_arr) in enumerate(qa_pairs):
                if hasattr(a_arr, "tolist"):
                    a_arr = a_arr.tolist()
                primary = a_arr[0] if isinstance(a_arr, list) and a_arr else str(a_arr)
                items.append({
                    "doc_id": doc_id,
                    "question": q,
                    "answer": primary,
                    "all_answers": a_arr if isinstance(a_arr, list) else [str(a_arr)],
                    "chunks": chunks,
                    "split": split,
                    "qi": qi,
                })
    return items


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _filter_superseded(memories: list, k: int) -> list:
    """Remove older memories that are near-duplicates of newer ones.

    For conflict resolution: if two memories are >0.85 cosine similar
    but have different timestamps, the older one is superseded by the
    newer one and should be removed.
    """
    if len(memories) <= 1:
        return memories[:k]

    superseded: set[str] = set()
    for i, a in enumerate(memories):
        if a.memory_id in superseded:
            continue
        a_emb = getattr(a, "embedding", None)
        a_ts = getattr(a, "timestamp", None)
        if not a_emb or not a_ts:
            continue

        for j in range(i + 1, len(memories)):
            b = memories[j]
            if b.memory_id in superseded:
                continue
            b_emb = getattr(b, "embedding", None)
            b_ts = getattr(b, "timestamp", None)
            if not b_emb or not b_ts:
                continue

            sim = _cosine_sim(a_emb, b_emb)
            if sim > 0.85 and a_ts != b_ts:
                # Keep the newer one, supersede the older
                if a_ts < b_ts:
                    superseded.add(a.memory_id)
                    break  # a is superseded, stop checking its pairs
                else:
                    superseded.add(b.memory_id)

    filtered = [m for m in memories if m.memory_id not in superseded]
    return filtered[:k]


# --- Runners ---

async def run_mnemotree(items: list[dict], *, store_dir: str, k: int,
                        generate_fn, judge_fn,
                        enable_decay: bool = False,
                        use_ppr: bool = False) -> tuple[list[CaseResult], dict]:
    results, timings = [], []
    ingested_docs: set[str] = set()
    cores: dict[str, Any] = {}

    # Use sqlite_vec when PPR is requested (needed for knowledge graph)
    backend = "sqlite_vec" if use_ppr else "chroma"

    for i, item in enumerate(items):
        doc_id = item["doc_id"]
        if doc_id not in cores:
            core = await create_memory_core(
                store_dir=Path(store_dir) / "mnemotree" / doc_id,
                sample_id=doc_id,
                enable_ner=True, enable_keywords=False, reranker="none",
                retrieval_mode="hybrid", enable_bm25=True,
                enable_decay=enable_decay,
                decay_stability_days=1.0,
                decay_floor=0.05,
                store_backend=backend,
            )
            cores[doc_id] = core

        if doc_id not in ingested_docs:
            from datetime import datetime, timedelta, timezone
            base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
            for ci, chunk in enumerate(item["chunks"]):
                # Give later chunks newer timestamps so decay favors them
                ts = base_time + timedelta(hours=ci)
                await cores[doc_id].remember(
                    content=chunk,
                    context={"doc_id": doc_id, "chunk_idx": ci},
                    analyze=False, summarize=False,
                    timestamp=ts,
                )
            ingested_docs.add(doc_id)
            print(f"    [mnemotree] Ingested {len(item['chunks'])} chunks for {doc_id}")

        t0 = time.perf_counter()
        # Retrieve more candidates, then filter conflicts
        fetch_k = k * 2 if enable_decay else k
        if use_ppr:
            # PPR: graph-augmented retrieval traverses entity links for multi-hop
            retrieved = await cores[doc_id].recall_ppr(item["question"], limit=fetch_k)
        else:
            retrieved = await cores[doc_id].recall(item["question"], limit=fetch_k, scoring=True, update_access=False)

        if enable_decay and len(retrieved) > 1:
            retrieved = _filter_superseded(retrieved, k)

        if enable_decay:
            # Sort by timestamp descending — newest facts first (primacy bias)
            retrieved = sorted(retrieved, key=lambda m: getattr(m, 'timestamp', None) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        context = "\n".join(f"[{j+1}] {m.content}" for j, m in enumerate(retrieved))

        predicted = ""
        if generate_fn:
            # Conflict-aware prompt for CR
            cr_prompt = (
                "You are answering questions using retrieved facts. "
                "IMPORTANT: The facts are ordered newest-first. If you see conflicting "
                "information about the same entity, ALWAYS use the version that appears "
                "FIRST (newest). Older facts may be outdated. Be concise and specific."
            ) if enable_decay else None
            predicted = await generate_fn(context, item["question"], system_prompt=cr_prompt)
        score = 0.0
        if judge_fn and predicted:
            score = await judge_fn(item["question"], item["answer"], predicted)
        timings.append((time.perf_counter() - t0) * 1000)

        results.append(CaseResult(
            case_id=f"{doc_id}_q{item['qi']}", question=item["question"],
            predicted=predicted, expected=item["answer"], score=score,
            category=item["split"],
        ))
        if (i + 1) % 10 == 0:
            print(f"    [mnemotree] {i+1}/{len(items)} done")

    return results, {"avg_case_ms": sum(timings)/len(timings) if timings else 0}


async def _run_baseline(cls, name, items, *, k, generate_fn, judge_fn, **kwargs):
    results, timings = [], []
    ingested_docs: set[str] = set()
    baselines: dict[str, Any] = {}

    for i, item in enumerate(items):
        doc_id = item["doc_id"]
        if doc_id not in baselines:
            baselines[doc_id] = cls(**kwargs)

        if doc_id not in ingested_docs:
            bl = baselines[doc_id]
            for ci, chunk in enumerate(item["chunks"]):
                await bl.remember(content=chunk, context={"doc_id": doc_id, "chunk_idx": ci})
            ingested_docs.add(doc_id)
            if i == 0 or (i + 1) % 10 == 0:
                print(f"    [{name}] Ingested {len(item['chunks'])} chunks for {doc_id}")

        t0 = time.perf_counter()
        retrieved = await baselines[doc_id].recall(item["question"], limit=k)
        context = "\n".join(f"[{j+1}] {m.content}" for j, m in enumerate(retrieved))

        predicted = ""
        if generate_fn:
            predicted = await generate_fn(context, item["question"])
        score = 0.0
        if judge_fn and predicted:
            score = await judge_fn(item["question"], item["answer"], predicted)
        timings.append((time.perf_counter() - t0) * 1000)

        results.append(CaseResult(
            case_id=f"{doc_id}_q{item['qi']}", question=item["question"],
            predicted=predicted, expected=item["answer"], score=score,
            category=item["split"],
        ))

    return results, {"avg_case_ms": sum(timings)/len(timings) if timings else 0}


def _aggregate(results):
    if not results:
        return {"accuracy": 0.0, "total": 0}
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category or "unknown", []).append(r.score)

    def _m(v): return sum(v)/len(v) if v else 0.0
    return {
        "accuracy": _m([r.score for r in results]),
        "total": len(results),
        "by_split": {c: {"accuracy": _m(s), "count": len(s)} for c, s in sorted(by_cat.items())},
    }


def _print_comparison(mt, ob, bl, mt_s, ob_s, bl_s):
    print(f"\n{'='*80}")
    print(f"  MEMORYAGENTBENCH: Mnemotree vs Obsidian vs Markdown")
    print(f"{'='*80}")
    print(f"  {'Metric':<30} {'Mnemotree':>10} {'Obsidian':>10} {'Markdown':>10}  {'MT-MD':>7} {'MT-OB':>7}")
    print(f"  {'-'*79}")
    ma, oa, ba = mt["accuracy"], ob["accuracy"], bl["accuracy"]
    print(f"  {'Overall accuracy':<30} {ma:>9.1%} {oa:>9.1%} {ba:>9.1%}  {ma-ba:>+6.1%} {ma-oa:>+6.1%}")
    all_cats = sorted(set(mt.get("by_split",{})) | set(ob.get("by_split",{})) | set(bl.get("by_split",{})))
    for cat in all_cats:
        mv = mt.get("by_split",{}).get(cat,{}).get("accuracy",0)
        ov = ob.get("by_split",{}).get(cat,{}).get("accuracy",0)
        bv = bl.get("by_split",{}).get(cat,{}).get("accuracy",0)
        n = mt.get("by_split",{}).get(cat,{}).get("count",0)
        print(f"  {cat:<30} {mv:>9.1%} {ov:>9.1%} {bv:>9.1%}  {mv-bv:>+6.1%} {mv-ov:>+6.1%}  (n={n})")
    print(f"\n  {'Avg latency (ms)':<30} {mt_s.get('avg_case_ms',0):>10.1f} {ob_s.get('avg_case_ms',0):>10.1f} {bl_s.get('avg_case_ms',0):>10.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="MemoryAgentBench: 3-way comparison")
    p.add_argument("--splits", default="ar,cr", help="Comma-separated splits to run")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--limit-docs", type=int, default=2, help="Max docs per split (0=all)")
    p.add_argument("--limit-questions", type=int, default=10, help="Max questions per doc (0=all)")
    p.add_argument("--store-dir", default=str(WORKSPACE_ROOT / "artifacts" / "stores" / "mab_comparison"))
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--output", default="benchmarks/results/mab_comparison.json")
    p.add_argument("--enable-decay", action="store_true", help="Enable temporal decay scoring (favor newer memories)")
    p.add_argument("--use-ppr", action="store_true", help="Use PPR graph retrieval with SQLiteVec backend")
    return p.parse_args()


async def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",")]

    print(f"Loading MemoryAgentBench splits: {splits}...")
    items = load_mab(splits, limit_docs=args.limit_docs, limit_questions=args.limit_questions)
    print(f"  {len(items)} total questions across {len(set(i['doc_id'] for i in items))} docs")
    from collections import Counter
    split_counts = Counter(i["split"] for i in items)
    print(f"  By split: {dict(split_counts)}")

    generate_fn = judge_fn = None
    if not args.no_judge:
        from benchmarks.lib.answering import generate_answer, judge_answer, get_openai_client
        client, model = get_openai_client()
        judge_model = args.judge_model
        async def generate_fn(ctx, q, system_prompt=None): return await generate_answer(client, model, ctx, q, system_prompt=system_prompt)
        async def judge_fn(q, exp, pred): return await judge_answer(client, judge_model, q, exp, pred)

    mode = "PPR" if args.use_ppr else ("decay" if args.enable_decay else "standard")
    print(f"\n[1/3] Running Mnemotree ({mode})...")
    mt_r, mt_s = await run_mnemotree(items, store_dir=args.store_dir, k=args.k, generate_fn=generate_fn, judge_fn=judge_fn, enable_decay=args.enable_decay or args.use_ppr, use_ppr=args.use_ppr)
    mt_sum = _aggregate(mt_r)
    print(f"  Mnemotree accuracy: {mt_sum['accuracy']:.1%}")

    print(f"\n[2/3] Running Obsidian...")
    ob_r, ob_s = await _run_baseline(ObsidianBaselineMemory, "obsidian", items, k=args.k, generate_fn=generate_fn, judge_fn=judge_fn, graph_boost=0.3, backlink_depth=1)
    ob_sum = _aggregate(ob_r)
    print(f"  Obsidian accuracy: {ob_sum['accuracy']:.1%}")

    print(f"\n[3/3] Running Markdown...")
    bl_r, bl_s = await _run_baseline(MarkdownBaselineMemory, "markdown", items, k=args.k, generate_fn=generate_fn, judge_fn=judge_fn, mode="tfidf")
    bl_sum = _aggregate(bl_r)
    print(f"  Markdown accuracy: {bl_sum['accuracy']:.1%}")

    _print_comparison(mt_sum, ob_sum, bl_sum, mt_s, ob_s, bl_s)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "memoryagentbench", "splits": splits,
        "num_cases": len(items), "k": args.k,
        "mnemotree": {"summary": mt_sum, "stats": mt_s},
        "obsidian_baseline": {"summary": ob_sum, "stats": ob_s},
        "markdown_baseline": {"summary": bl_sum, "stats": bl_s},
    }
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
