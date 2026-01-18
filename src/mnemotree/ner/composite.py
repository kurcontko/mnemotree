from __future__ import annotations

import asyncio

from .base import BaseNER, NERResult


class CompositeNER(BaseNER):
    """Combines multiple NER implementations."""

    def __init__(self, implementations: list[tuple[BaseNER, float]]):
        """
        Initialize CompositeNER.

        Args:
            implementations: List of (implementation, weight) tuples
        """
        self.implementations = implementations

    async def extract_entities(self, text: str) -> NERResult:
        """Combine results from multiple implementations."""
        # Get results from all implementations
        results = await asyncio.gather(
            *[impl.extract_entities(text) for impl, _ in self.implementations]
        )

        # Combine entities and mentions with weighted confidence
        combined_entities = {}
        combined_mentions = {}
        combined_confidence = {}

        for result, (_, weight) in zip(results, self.implementations, strict=True):
            for entity, entity_type in result.entities.items():
                if entity not in combined_entities:
                    combined_entities[entity] = entity_type
                    combined_mentions[entity] = result.mentions.get(entity, [])
                    combined_confidence[entity] = (
                        result.confidence.get(entity, 1.0) * weight if result.confidence else weight
                    )
                else:
                    # Combine mentions
                    combined_mentions[entity].extend(
                        mention
                        for mention in result.mentions.get(entity, [])
                        if mention not in combined_mentions[entity]
                    )

                    # Update confidence with weighted average
                    if result.confidence:
                        current_conf = combined_confidence[entity]
                        new_conf = result.confidence.get(entity, 1.0) * weight
                        combined_confidence[entity] = (current_conf + new_conf) / 2

        return NERResult(
            entities=combined_entities, mentions=combined_mentions, confidence=combined_confidence
        )
