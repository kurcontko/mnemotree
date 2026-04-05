"""SCMRAG-style gap detection for iterative retrieval correction.

After an initial retrieval pass the GapAnalyzer identifies what information
is missing and returns targeted sub-queries to fill those gaps.

Two implementations are provided:
- HeuristicGapAnalyzer: lightweight, no LLM required (lite mode)
- LLMGapAnalyzer: uses an OpenAI-compatible endpoint (pro mode)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import MemoryItem

logger = logging.getLogger(__name__)

# Wh-question words mapped to the type of entity expected in the answer
_WH_ENTITY_TYPES: dict[str, list[str]] = {
    "who": ["PERSON", "ORG"],
    "whose": ["PERSON", "ORG"],
    "whom": ["PERSON", "ORG"],
    "where": ["LOC", "GPE", "FAC"],
    "when": ["DATE", "TIME", "EVENT"],
    "what": [],  # open-ended
    "which": [],
    "how": [],
    "why": [],
}


@runtime_checkable
class GapAnalyzer(Protocol):
    """Protocol for gap analyzers."""

    async def detect_gaps(
        self,
        query: str,
        retrieved: list[MemoryItem],
        query_entities: dict[str, str],
    ) -> list[str]:
        """Return targeted sub-queries for detected information gaps.

        Args:
            query: The original user query.
            retrieved: Memories already retrieved.
            query_entities: entity_text → entity_type from NER on the query.

        Returns:
            List of sub-query strings (empty if no gaps detected).
        """
        ...


class HeuristicGapAnalyzer:
    """Lightweight gap detection without an LLM.

    Strategy:
    1. Find entities mentioned in the query that are absent from retrieved memories.
    2. For wh-questions, check whether the expected answer-type entity is present.
    3. Return one sub-query per gap.
    """

    def __init__(self, max_gaps: int = 3) -> None:
        self.max_gaps = max_gaps

    async def detect_gaps(
        self,
        query: str,
        retrieved: list[MemoryItem],
        query_entities: dict[str, str],
    ) -> list[str]:
        sub_queries: list[str] = []

        # Collect all entity mentions from retrieved memories
        covered_entities: set[str] = set()
        for memory in retrieved:
            covered_entities.update(e.lower() for e in (memory.entities or {}))

        # Gap 1: query entities missing from retrieved content
        for entity_text, _entity_type in query_entities.items():
            if entity_text.lower() not in covered_entities:
                sub_queries.append(f"Tell me about {entity_text}")
            if len(sub_queries) >= self.max_gaps:
                break

        # Gap 2: wh-question answer type missing
        if len(sub_queries) < self.max_gaps:
            wh_word = _extract_wh_word(query)
            if wh_word:
                expected_types = _WH_ENTITY_TYPES.get(wh_word.lower(), [])
                if expected_types:
                    covered_types = set()
                    for memory in retrieved:
                        for _e, etype in (memory.entities or {}).items():
                            if etype:
                                covered_types.add(etype.upper())
                    if not any(t in covered_types for t in expected_types):
                        sub_queries.append(f"{wh_word.capitalize()} is related to: {query}")

        return sub_queries[: self.max_gaps]


class LLMGapAnalyzer:
    """LLM-powered gap detection using an OpenAI-compatible endpoint.

    Sends a structured prompt to the LLM describing the original query and
    retrieved memories, then parses the response into sub-queries.
    """

    _SYSTEM_PROMPT = (
        "You are a memory retrieval assistant. "
        "Given an original query and already-retrieved memories, "
        "identify at most {max_gaps} specific pieces of information that are MISSING. "
        "For each gap, output exactly one concise retrieval sub-query per line. "
        "If retrieval is complete, output only: COMPLETE"
    )

    _USER_PROMPT = (
        "Original query: {query}\n\n"
        "Retrieved memories ({n} total):\n{summaries}\n\n"
        "What key information is still missing?"
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str = "",
        max_tokens: int = 256,
        max_gaps: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_gaps = max_gaps

    async def detect_gaps(
        self,
        query: str,
        retrieved: list[MemoryItem],
        query_entities: dict[str, str],
    ) -> list[str]:
        try:
            from openai import AsyncOpenAI
        except ImportError as err:
            raise ImportError(
                "LLMGapAnalyzer requires openai. Install with `pip install openai`."
            ) from err

        import os

        api_key = (
            self.api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )

        client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)

        summaries = "\n".join(
            f"- [{i + 1}] {m.content[:200]}" for i, m in enumerate(retrieved[:10])
        )

        messages = [
            {
                "role": "system",
                "content": self._SYSTEM_PROMPT.format(max_gaps=self.max_gaps),
            },
            {
                "role": "user",
                "content": self._USER_PROMPT.format(
                    query=query,
                    n=len(retrieved),
                    summaries=summaries or "(none)",
                ),
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("LLMGapAnalyzer LLM call failed")
            return []

        if "COMPLETE" in text.upper():
            return []

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Strip leading bullets / numbering
        sub_queries = [re.sub(r"^[\-\*\d\.\)]+\s*", "", ln) for ln in lines]
        return [sq for sq in sub_queries if sq][: self.max_gaps]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_wh_word(text: str) -> str | None:
    """Return the first wh-question word found in text, or None."""
    for word in text.lower().split():
        word_clean = re.sub(r"[^a-z]", "", word)
        if word_clean in _WH_ENTITY_TYPES:
            return word_clean
    return None
