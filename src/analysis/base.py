
from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import json
from typing import Dict

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class AnalysisStrategy(Enum):
    """Defines different strategies for memory analysis"""
    QUICK = "quick"  # Fast, surface-level analysis
    DEEP = "deep"    # Thorough, detailed analysis
    ADAPTIVE = "adaptive"  # Adjusts based on content complexity


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.chain = self._create_chain()
    
    @abstractmethod
    def _get_template(self) -> str:
        """Return the prompt template for the analyzer."""
        pass
    
    @abstractmethod
    def _get_parser(self) -> BaseOutputParser:
        """Return the output parser for the analyzer."""
        pass
    
    def _create_chain(self) -> Runnable:
        """Create the analysis chain."""
        parser = self._get_parser()
        template = self._get_template()
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt | self.llm | parser
    
    async def analyze(self, content: str, context_str: str) -> Dict:
        """Perform analysis using the chain."""
        return await self.chain.ainvoke({
            "content": content,
            "context": context_str
        })

