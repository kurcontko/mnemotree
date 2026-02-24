"""Abstract base class for content normalizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseNormalizer(ABC):
    """Base class for content normalizers.

    Normalizers transform raw memory content before storage to improve
    retrieval quality (e.g., resolving pronouns, anchoring relative dates).
    """

    @abstractmethod
    async def normalize(self, content: str, context: dict[str, Any] | None = None) -> str:
        """Normalize content text.

        Args:
            content: Raw content string to normalize.
            context: Optional context dict with metadata such as speaker names,
                     reference dates, recent entities, etc.

        Returns:
            Normalized content string.
        """
        ...
