from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_COMPOUND_MARKERS = [
    "additionally",
    "furthermore",
    "however",
    "also",
    "moreover",
    "on the other hand",
    "in contrast",
    "meanwhile",
]


class FactDecomposer:
    """Splits compound content into atomic independent facts using an LLM."""

    def __init__(self, llm: Any, max_facts: int = 10) -> None:
        self.llm = llm
        self.max_facts = max_facts

    def is_compound(self, content: str, min_length: int) -> bool:
        """Return True if content appears to contain multiple independent claims."""
        if len(content) < min_length:
            return False
        sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
        if len(sentences) >= 3:
            return True
        lower = content.lower()
        return any(marker in lower for marker in _COMPOUND_MARKERS)

    async def decompose(self, content: str) -> list[str]:
        """Decompose content into atomic facts. Falls back to [content] on failure."""
        prompt = (
            "Decompose the following text into a list of atomic, independent facts. "
            "Each fact should be self-contained and not require other facts to be understood. "
            "Reply with valid JSON only, no markdown fences: "
            '{"facts": ["fact 1", "fact 2", ...]}\n\n'
            f'Text: """{content}"""'
        )
        try:
            response = await self.llm.ainvoke(prompt)
            import json

            text = (response.content if hasattr(response, "content") else str(response)) or ""
            data = json.loads(text)
            raw_facts = data.get("facts", []) if isinstance(data, dict) else []
            facts: list[str] = [str(f) for f in raw_facts if str(f).strip()]
            if not facts:
                return [content]
            return facts[: self.max_facts]
        except Exception:
            logger.debug("Fact decomposition failed", exc_info=True)
            return [content]
