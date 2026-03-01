"""SimpleMem-style intent classification for pre-filtering retrieval by MemoryType.

Reference: arXiv:2601.02553 — infer retrieval intent from the query to narrow
the vector search space before embedding lookup.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .models import MemoryType

if TYPE_CHECKING:
    from langchain_core.language_models.base import BaseLanguageModel

__all__ = [
    "RetrievalIntent",
    "INTENT_TO_TYPES",
    "IntentClassifier",
    "KeywordIntentClassifier",
    "LLMIntentClassifier",
]


class RetrievalIntent(str, Enum):
    EPISODIC = "episodic"    # "What happened when...", "When did I..."
    SEMANTIC = "semantic"    # "What is...", "How does...", "Why is..."
    PROCEDURAL = "procedural"  # "How do I...", "Steps to...", "How to..."
    UNKNOWN = "unknown"


INTENT_TO_TYPES: dict[RetrievalIntent, list[MemoryType]] = {
    RetrievalIntent.EPISODIC: [MemoryType.EPISODIC, MemoryType.AUTOBIOGRAPHICAL],
    RetrievalIntent.SEMANTIC: [MemoryType.SEMANTIC],
    RetrievalIntent.PROCEDURAL: [MemoryType.PROCEDURAL, MemoryType.PRIMING],
}


class IntentClassifier(Protocol):
    async def classify(self, query: str) -> RetrievalIntent: ...


class KeywordIntentClassifier:
    """Heuristic, zero-inference classifier. Works in lite mode without an LLM."""

    EPISODIC_TRIGGERS: frozenset[str] = frozenset(
        {"when", "last time", "what happened", "remember", "yesterday", "ago", "did i", "have i"}
    )
    SEMANTIC_TRIGGERS: frozenset[str] = frozenset(
        {"what is", "what are", "why is", "how does", "explain", "define", "what's"}
    )
    PROCEDURAL_TRIGGERS: frozenset[str] = frozenset(
        {"how to", "how do i", "steps", "procedure", "configure", "set up", "setup", "install"}
    )

    async def classify(self, query: str) -> RetrievalIntent:
        q = query.lower()
        proc = sum(1 for t in self.PROCEDURAL_TRIGGERS if t in q)
        sem = sum(1 for t in self.SEMANTIC_TRIGGERS if t in q)
        ep = sum(1 for t in self.EPISODIC_TRIGGERS if t in q)

        best = max(proc, sem, ep)
        if best == 0:
            return RetrievalIntent.UNKNOWN
        if proc == best:
            return RetrievalIntent.PROCEDURAL
        if sem == best:
            return RetrievalIntent.SEMANTIC
        return RetrievalIntent.EPISODIC


_CLASSIFY_PROMPT = """\
Classify the retrieval intent of this query as exactly one of:
  episodic   - personal experiences or events ("when did...", "what happened when...")
  semantic   - facts or knowledge ("what is...", "why is...", "explain...")
  procedural - skills or steps ("how to...", "how do I...", "configure...")
  unknown    - unclear or mixed intent

Query: {query}
Intent (one word):""".strip()


class LLMIntentClassifier:
    """Few-shot LLM-based classifier. Requires an LLM (pro mode)."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm

    async def classify(self, query: str) -> RetrievalIntent:
        response = await self.llm.ainvoke(_CLASSIFY_PROMPT.format(query=query))
        text = (
            response.content if hasattr(response, "content") else str(response)
        ).strip().lower()
        for intent in RetrievalIntent:
            if intent.value in text:
                return intent
        return RetrievalIntent.UNKNOWN
