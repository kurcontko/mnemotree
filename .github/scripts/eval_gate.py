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


def check_floors(store: str, scores: dict[str, float], floors: dict[str, float]) -> list[str]:
    return [
        f"{store}: {name} {scores[name]:.4f} < floor {floor}"
        for name, floor in floors.items()
        if scores[name] < floor
    ]


def check_spread(scores: dict[str, dict[str, float]], names: list[str], limit: float) -> list[str]:
    failures = []
    for name in names:
        values = {store: s[name] for store, s in scores.items()}
        spread = max(values.values()) - min(values.values())
        if spread > limit:
            detail = ", ".join(f"{s}={v:.4f}" for s, v in values.items())
            failures.append(f"stores disagree on {name} by {spread:.4f}: {detail}")
    return failures


def table(scores: dict[str, dict[str, float]], floors: dict[str, float]) -> str:
    rows = [
        "| Store | " + " | ".join(floors) + " |",
        "| --- |" + " --- |" * len(floors),
        *(
            f"| {store} | " + " | ".join(f"{s[name]:.4f}" for name in floors) + " |"
            for store, s in scores.items()
        ),
        "| floor | " + " | ".join(str(v) for v in floors.values()) + " |",
    ]
    return "\n".join(rows)


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
        failures += check_floors(store, scores[store], floors)

    if len(scores) > 1:
        failures += check_spread(scores, list(floors), config["max_store_spread"])

    print(table(scores, floors))
    for failure in failures:
        print(f"::error title=eval gate::{failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
