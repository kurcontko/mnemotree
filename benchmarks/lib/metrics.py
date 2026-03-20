"""Retrieval and QA metrics shared across benchmark adapters.

Extracted from evaluate.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    denom = min(k, len(retrieved)) if retrieved else k
    if denom == 0:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / denom


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in retrieved[:k] if item in relevant) / len(relevant)


def mrr_score(retrieved: Sequence[str], relevant: set[str]) -> float:
    for idx, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rels = [1.0 if item in relevant else 0.0 for item in retrieved[:k]]
    if not rels:
        return 0.0
    dcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(rels, 1))
    idcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(sorted(rels, reverse=True), 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def session_recall(retrieved_session_ids: set[str], expected_session_ids: set[str]) -> float:
    """Fraction of expected sessions that appear in retrieved memories."""
    if not expected_session_ids:
        return 0.0
    return len(retrieved_session_ids & expected_session_ids) / len(expected_session_ids)
