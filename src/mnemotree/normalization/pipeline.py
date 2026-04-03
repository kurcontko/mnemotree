"""Normalization pipeline that chains multiple normalizers."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseNormalizer

logger = logging.getLogger(__name__)


class NormalizationPipeline:
    """Sequential chain of normalizers applied to content before storage.

    Each normalizer transforms the content in order, passing the result
    to the next normalizer in the chain.
    """

    def __init__(self, normalizers: list[BaseNormalizer]) -> None:
        self.normalizers = normalizers

    async def normalize(self, content: str, context: dict[str, Any] | None = None) -> str:
        """Run all normalizers sequentially on the content.

        Args:
            content: Raw content string.
            context: Optional context dict passed to each normalizer.

        Returns:
            Normalized content after all normalizers have been applied.
        """
        for normalizer in self.normalizers:
            try:
                content = await normalizer.normalize(content, context)
            except Exception:
                logger.debug(
                    "Normalizer %s failed, skipping",
                    type(normalizer).__name__,
                    exc_info=True,
                )
        return content

    def __len__(self) -> int:
        return len(self.normalizers)

    def __bool__(self) -> bool:
        return len(self.normalizers) > 0
