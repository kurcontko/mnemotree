from __future__ import annotations

import asyncio

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

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
For each entity, specify its type.

Text: {text}

{format_instructions}
Provide confidence scores between 0 and 1 for each entity.
"""
    
    async def extract_entities(self, text: str) -> NERResult:
        """Extract entities using LLM."""
        # Prepare prompt
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["text", "text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        chain = prompt | self.llm | self.parser
        
        # Get LLM response
        response = await chain.ainvoke({"text": text})
        
        try:
            # Parse JSON response
            result = response.json()
            entities = result.get("entities", [])
            confidence = result.get("confidence", {})
            
            # Extract mentions (contexts) for each entity
            mentions = {}
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
            
            return NERResult(
                entities=entities,
                mentions=mentions,
                confidence=confidence
            )
        except Exception as e:
            # Fallback to empty result on parsing error
            print(f"Error parsing LLM response: {e}")
            return NERResult(entities={}, mentions={})
