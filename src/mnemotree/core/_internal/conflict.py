from __future__ import annotations

import logging
from typing import Any

from ..models import MemoryItem

logger = logging.getLogger(__name__)

_NEGATION_WORDS = frozenset(
    [
        "not",
        "never",
        "no",
        "isn't",
        "aren't",
        "doesn't",
        "don't",
        "won't",
        "can't",
        "false",
        "wrong",
        "incorrect",
    ]
)


class ConflictDetector:
    """Detects semantic conflicts between memories using heuristics and optional LLM verification."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    async def detect(
        self,
        new_memory: MemoryItem,
        candidates: list[MemoryItem],
    ) -> list[tuple[MemoryItem, str]]:
        """Return list of (conflicting_memory, reason) pairs."""
        results: list[tuple[MemoryItem, str]] = []
        for candidate in candidates:
            if not self._heuristic_conflict(new_memory, candidate):
                continue
            reason = "heuristic: negation + shared entities"
            if self.llm:
                llm_result = await self._llm_conflict(new_memory.content, candidate.content)
                if not llm_result["contradicts"]:
                    continue
                reason = llm_result["reason"] or reason
            results.append((candidate, reason))
        return results

    def _heuristic_conflict(self, a: MemoryItem, b: MemoryItem) -> bool:
        shared = set(a.entities or {}) & set(b.entities or {})
        if not shared:
            return False
        combined = (a.content + " " + b.content).lower().split()
        return bool(_NEGATION_WORDS & set(combined))

    async def _llm_conflict(self, content_a: str, content_b: str) -> dict:
        prompt = (
            "Do these two memories contradict each other? "
            "Reply with valid JSON only, no markdown fences: "
            '{"contradicts": true/false, "reason": "brief explanation"}\n\n'
            f'Memory A: """{content_a}"""\n'
            f'Memory B: """{content_b}"""'
        )
        try:
            response = await self.llm.ainvoke(prompt)  # type: ignore[union-attr]
            import json

            text = response.content if hasattr(response, "content") else str(response)
            data = json.loads(text)
            return {
                "contradicts": bool(data.get("contradicts", False)),
                "reason": str(data.get("reason", "")),
            }
        except Exception:
            logger.debug("LLM conflict check failed", exc_info=True)
            return {"contradicts": False, "reason": ""}
