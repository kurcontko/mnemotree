from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["summary"]["metrics"]


def _as_float_map(d: dict[str, Any]) -> dict[int, float]:
    return {int(k): float(v) for k, v in d.items()}


def _row(path: Path) -> dict[str, Any]:
    doc = _load(path)
    cfg = doc.get("config", {})
    m = _metrics(doc)
    return {
        "path": str(path),
        "store": cfg.get("store"),
        "retrieval_mode": cfg.get("retrieval_mode"),
        "enable_ner": cfg.get("enable_ner"),
        "enable_bm25": cfg.get("enable_bm25"),
        "mrr": float(m["mrr"]),
        "semantic_similarity": float(m.get("semantic_similarity") or 0.0),
        "precision": _as_float_map(m["precision@k"]),
        "recall": _as_float_map(m["recall@k"]),
        "ndcg": _as_float_map(m["ndcg@k"]),
    }


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _delta(x: float, baseline: float) -> str:
    d = x - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _print_table(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    k_values: list[int],
) -> None:
    header = [
        "file",
        "store",
        "ner",
        "bm25",
        "mrr",
        "Δmrr",
    ]
    for k in k_values:
        header += [f"p@{k}", f"Δp@{k}", f"r@{k}", f"Δr@{k}", f"n@{k}", f"Δn@{k}"]

    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")

    def emit(row: dict[str, Any]) -> None:
        vals = [
            Path(row["path"]).name,
            str(row.get("store")),
            str(bool(row.get("enable_ner"))),
            str(bool(row.get("enable_bm25"))),
            _fmt(row["mrr"]),
            _delta(row["mrr"], baseline["mrr"]),
        ]
        for k in k_values:
            vals += [
                _fmt(row["precision"].get(k, 0.0)),
                _delta(row["precision"].get(k, 0.0), baseline["precision"].get(k, 0.0)),
                _fmt(row["recall"].get(k, 0.0)),
                _delta(row["recall"].get(k, 0.0), baseline["recall"].get(k, 0.0)),
                _fmt(row["ndcg"].get(k, 0.0)),
                _delta(row["ndcg"].get(k, 0.0), baseline["ndcg"].get(k, 0.0)),
            ]
        print("| " + " | ".join(vals) + " |")

    emit(baseline)
    for row in candidates:
        emit(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark result JSON files.")
    parser.add_argument("baseline", help="Baseline result JSON path")
    parser.add_argument("candidates", nargs="+", help="Candidate result JSON paths")
    parser.add_argument(
        "--k",
        default="1,3,5,10",
        help="Comma-separated k values to include (default: 1,3,5,10)",
    )
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    baseline_row = _row(Path(args.baseline))
    candidate_rows = [_row(Path(p)) for p in args.candidates]

    _print_table(baseline_row, candidate_rows, k_values)


if __name__ == "__main__":
    main()
