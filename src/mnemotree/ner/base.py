from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class NERResult(BaseModel):
    """Structured output from NER processing."""

    entities: dict[str, str]  # Entity text -> Entity type
    mentions: dict[str, list[str]]  # Entity text -> List of context snippets
    confidence: dict[str, float] | None = None  # Entity text -> Confidence score


class BaseNER(ABC):
    """Abstract base class for NER implementations."""

    @abstractmethod
    async def extract_entities(self, text: str) -> NERResult:
        """
        Extract named entities from text.

        Args:
            text: Input text to process

        Returns:
            NERResult containing extracted entities and their mentions
        """

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for an entity mention."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
