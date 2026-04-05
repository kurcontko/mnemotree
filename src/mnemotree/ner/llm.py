from __future__ import annotations

import logging
from typing import Any

from .base import BaseNER, NERResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Extract named entities from the provided text.
Return a JSON object with:
- "entities": dict mapping entity text to its type (e.g., "New York": "Location")
- "mentions": dict mapping entity text to list of context snippets
- "confidence": dict mapping entity text to a confidence float 0-1

Use flat dicts. Entity types: Person, Organization, Location, Date, Event, Product, Other."""


class PydanticAILLMNER(BaseNER):
    """PydanticAI LLM-based NER (replaces LangchainLLMNER)."""

    def __init__(self, model: str | Any = "openai:gpt-4o-mini"):
        self._model = model
        self._agent: Any = None

    def _ensure_agent(self) -> None:
        if self._agent is None:
            try:
                from pydantic_ai import Agent
            except ImportError as exc:
                raise RuntimeError(
                    "pydantic-ai is required for LLM-based NER. "
                    "Install mnemotree with the 'analysis' extra."
                ) from exc
            self._agent = Agent(self._model, output_type=NERResult, instructions=_SYSTEM_PROMPT)

    async def extract_entities(self, text: str) -> NERResult:
        """Extract entities using LLM."""
        try:
            self._ensure_agent()
            result = await self._agent.run(f"Text: {text}")
            ner = result.output
            confidence = ner.confidence or {}

            mentions: dict[str, list[str]] = {}
            for entity in ner.entities:
                ctxs: list[str] = []
                start = 0
                while (pos := text.find(entity, start)) != -1:
                    ctxs.append(self._get_context(text, pos, pos + len(entity)))
                    start = pos + 1
                mentions[entity] = ctxs

            return NERResult(entities=ner.entities, mentions=mentions, confidence=confidence)
        except Exception as e:
            logger.warning("Error in PydanticAI NER", exc_info=e)
            return NERResult(entities={}, mentions={})


# Backward-compat alias
LangchainLLMNER = PydanticAILLMNER
