from __future__ import annotations

from typing import Any

from .base import BaseNER, NERResult


class GLiNERNER(BaseNER):
    """GLiNER-based NER implementation for domain-specific entities."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_medium-v2.1",
        entity_types: list[str] | None = None,
        threshold: float = 0.3,
    ):
        """
        Initialize GLiNER NER.

        Args:
            model_name: GLiNER model name from HuggingFace
            entity_types: List of entity types to extract (e.g., ['person', 'location', 'dish', 'ingredient'])
            threshold: Confidence threshold for entity extraction
        """
        try:
            from gliner import GLiNER
        except ImportError as err:
            raise ImportError("GLiNER is not installed. Install with: pip install gliner") from err

        self.model = GLiNER.from_pretrained(model_name)
        self.entity_types = entity_types or [
            "person",
            "location",
            "organization",
            "dish",
            "ingredient",
            "cuisine",
            "food",
        ]
        self.threshold = threshold

    async def extract_entities(self, text: str) -> NERResult:
        """Extract entities using GLiNER."""
        # GLiNER is synchronous, so run it directly
        predictions = self.model.predict_entities(text, self.entity_types, threshold=self.threshold)

        entities: dict[str, str] = {}
        mentions: dict[str, list[str]] = {}
        confidence: dict[str, float] = {}

        for pred in predictions:
            entity_text = pred["text"]
            entity_type = pred["label"]
            score = pred["score"]

            # Store entity with its type
            entities[entity_text] = entity_type

            # Store confidence
            confidence[entity_text] = score

            # Store mention context
            if entity_text not in mentions:
                mentions[entity_text] = []

            # Get context around the entity
            start_idx = pred.get("start", 0)
            end_idx = pred.get("end", len(entity_text))
            context = self._get_context(text, start_idx, end_idx)
            mentions[entity_text].append(context)

        return NERResult(entities=entities, mentions=mentions, confidence=confidence)

