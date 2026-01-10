from __future__ import annotations

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from .base import BaseNER, NERResult


class LangchainLLMNER(BaseNER):
    """LangChain LLM-based NER implementation."""

    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize LangChainNER.

        Args:
            llm: LangChain language model to use
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=NERResult)
        self.prompt_template = """Extract named entities from the following text.
Return a JSON object with this exact structure:
{{
  "entities": {{"entity_text": "entity_type", ...}},
  "mentions": {{"entity_text": ["context snippet", ...], ...}},
  "confidence": {{"entity_text": 0.95, ...}}
}}

Important:
- The "entities" field maps entity text to its type (e.g., "New York": "Location")
- The "confidence" field maps entity text to a single float score (e.g., "New York": 0.95)
- Use flat dictionaries, not nested by entity type
- Confidence scores should be between 0 and 1

Text: {text}

{format_instructions}
"""

    async def extract_entities(self, text: str) -> NERResult:
        """Extract entities using LLM."""
        # Prepare prompt
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.parser

        try:
            parsed = await chain.ainvoke({"text": text})
            result = (
                parsed
                if isinstance(parsed, dict)
                else getattr(parsed, "model_dump", lambda: parsed)()
            )
            ner_result = NERResult.model_validate(result)
            entities = ner_result.entities
            confidence = ner_result.confidence or {}

            # Extract mentions (contexts) for each entity
            mentions: dict[str, list[str]] = {}
            for entity in entities:
                # Simple string matching for mentions
                # Could be improved with more sophisticated matching
                start = 0
                entity_mentions = []
                while True:
                    pos = text.find(entity, start)
                    if pos == -1:
                        break
                    context = self._get_context(text, pos, pos + len(entity))
                    entity_mentions.append(context)
                    start = pos + 1
                mentions[entity] = entity_mentions

            return NERResult(entities=entities, mentions=mentions, confidence=confidence)
        except (ValidationError, TypeError, ValueError) as e:
            # Fallback to empty result on parsing error
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Error parsing LLM response", exc_info=e)
            return NERResult(entities={}, mentions={})
