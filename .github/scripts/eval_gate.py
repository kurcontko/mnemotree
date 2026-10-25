"""Gate offline retrieval quality on benchmarks/evaluate.py results.

Reads the [eval] section of budget.toml and one evaluate.py result JSON per
store. Fails when a store is missing, when any metric falls under its floor,
or when stores disagree with each other: with identical memories and
identical deterministic embeddings, every store must rank the same way, so a
spread between them is a store bug, not noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


def metric(summary: dict, name: str) -> float:
    """`recall@10` → summary["metrics"]["recall@k"]["10"]; `mrr` → ["mrr"]."""
    metrics = summary["metrics"]
    if "@" in name:
        family, k = name.split("@", 1)
        return float(metrics[f"{family}@k"][k])
    return float(metrics[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory holding <store>.json files")
    parser.add_argument("--budget", type=Path, default=Path("budget.toml"))
    args = parser.parse_args()

    config = tomllib.loads(args.budget.read_text())["eval"]
    floors: dict[str, float] = config["floors"]
    scores: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for store in config["stores"]:
        path = args.results / f"{store}.json"
        if not path.is_file():
            failures.append(f"{store}: no result file (did the evaluation crash?)")
            continue
        summary = json.loads(path.read_text())["summary"]
        scores[store] = {name: metric(summary, name) for name in floors}
        for name, floor in floors.items():
            if scores[store][name] < floor:
                failures.append(f"{store}: {name} {scores[store][name]:.4f} < floor {floor}")

    if len(scores) > 1:
        for name in floors:
            values = {store: s[name] for store, s in scores.items()}
            spread = max(values.values()) - min(values.values())
            if spread > config["max_store_spread"]:
                detail = ", ".join(f"{s}={v:.4f}" for s, v in values.items())
                failures.append(f"stores disagree on {name} by {spread:.4f}: {detail}")

    header = "| Store | " + " | ".join(floors) + " |"
    rule = "| --- |" + " --- |" * len(floors)
    rows = [
        f"| {store} | " + " | ".join(f"{s[name]:.4f}" for name in floors) + " |"
        for store, s in scores.items()
    ]
    floor_row = "| floor | " + " | ".join(str(v) for v in floors.values()) + " |"
    print("\n".join([header, rule, *rows, floor_row]))
    for failure in failures:
        print(f"::error title=eval gate::{failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
