"""Standardized result schema for all benchmark outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import CaseResult


def build_result(
    *,
    benchmark: str,
    split: str,
    summary: dict[str, Any],
    per_case_results: list[CaseResult],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the standard result JSON envelope."""
    return {
        "benchmark": benchmark,
        "split": split,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "per_case_results": [
            {
                "case_id": r.case_id,
                "question": r.question,
                "predicted": r.predicted,
                "expected": r.expected,
                "score": r.score,
                "category": r.category,
                "retrieved_ids": r.retrieved_ids,
                "metadata": r.metadata,
            }
            for r in per_case_results
        ],
        "config": config,
    }


def save_result(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write result JSON to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    return path
