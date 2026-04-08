"""MS-RAG style multi-step query decomposition for compound queries.

Decomposes a compound query into independent sub-questions, retrieves each in
parallel, and fuses results with RRF deduplication.

Reference: EMNLP 2025 — Multi-Step RAG, +10–30% on compound queries.
"""

from __future__ import annotations

from typing import Any

from .models import MemoryItem
from .retrieval import rrf_fuse

__all__ = ["QueryDecomposer", "rrf_merge"]

_DECOMPOSE_PROMPT = """\
Decompose the following question into up to {max_sub} simple, independent sub-questions
that together answer the original. If the question is already simple, return only the
original. Output one sub-question per line, no numbering, no extra text.
Question: {query}""".strip()


class QueryDecomposer:
    """Decomposes a compound query into independent sub-queries via LLM."""

    def __init__(self, llm: Any, max_sub_queries: int = 4) -> None:
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    async def decompose(self, query: str) -> list[str]:
        """Return [query] if already simple, else [q1, q2, ...] (up to max_sub_queries)."""
        response = await self.llm.ainvoke(
            _DECOMPOSE_PROMPT.format(query=query, max_sub=self.max_sub_queries)
        )
        content = (response.content if hasattr(response, "content") else str(response)) or ""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return lines[: self.max_sub_queries] or [query]


def rrf_merge(
    results_per_query: list[list[MemoryItem]],
    limit: int | None,
) -> list[MemoryItem]:
    """RRF-fuse memories from multiple sub-query retrievals, dedup by memory_id."""
    stage_candidates = dict(enumerate(results_per_query))
    fused, _scores, _stage_scores = rrf_fuse(stage_candidates=stage_candidates)
    if limit is not None:
        return fused[:limit]
    return fused
