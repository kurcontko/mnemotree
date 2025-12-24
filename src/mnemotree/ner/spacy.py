from __future__ import annotations

import asyncio

import spacy

from .base import BaseNER, NERResult


class SpacyNER(BaseNER):
    """spaCy-based NER implementation."""
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize SpacyNER.
        
        Args:
            model: Name of spaCy model to use
        """
        self.nlp = spacy.load(model)

    async def extract_entities(self, text: str) -> NERResult:
        """Extract entities using spaCy."""
        # Process using spaCy in a thread pool to avoid blocking
        doc = await asyncio.to_thread(self.nlp, text)
        
        entities = {}
        mentions = {}
        
        for ent in doc.ents:
            # Store entity and type
            entities[ent.text] = ent.label_
            
            # Get and store context
            context = self._get_context(text, ent.start_char, ent.end_char)
            if ent.text not in mentions:
                mentions[ent.text] = []
            mentions[ent.text].append(context)
        
        return NERResult(
            entities=entities,
            mentions=mentions
        )
