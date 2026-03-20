from __future__ import annotations

import asyncio

try:
    import spacy
except ImportError:
    spacy = None

from .base import BaseNER, NERResult


class SpacyNER(BaseNER):
    """spaCy-based NER implementation."""

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize SpacyNER.

        Args:
            model: Name of spaCy model to use
        """
        if spacy is None:
            raise ImportError("SpacyNER requires spacy. Install with `pip install spacy`.")
        self.nlp = spacy.load(model)

    async def extract_entities(self, text: str) -> NERResult:
        """Extract entities using spaCy."""
        # Process using spaCy in a thread pool to avoid blocking
        doc = await asyncio.to_thread(self.nlp, text)

        entities: dict[str, str] = {}
        mentions: dict[str, list[str]] = {}

        for ent in doc.ents:
            # Store entity and type
            entities[ent.text] = ent.label_

            # Get and store context snippet
            context = self._get_context(text, ent.start_char, ent.end_char)
            mentions.setdefault(ent.text, []).append(context)

        return NERResult(entities=entities, mentions=mentions)
