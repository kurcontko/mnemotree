from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
from typing import Dict, List, Optional, Any

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from ..core.models import MemoryType
from .models import (
    MemoryAnalysisResult,
    MemoryClassificationResult,
    EmotionAnalysisResult,
    ConceptExtractionResult,
    InsightsResult
)
from .base import BaseAnalyzer


class MemoryClassifierAnalyzer(BaseAnalyzer):
    """Handles memory classification analysis."""
    
    def _get_parser(self) -> JsonOutputParser:
        return JsonOutputParser(pydantic_object=MemoryClassificationResult)
    
    def _get_template(self) -> str:
        return """
Analyze the following conversation or content and classify it:

Content: {content}
Context: {context}

Classify the memory type and importance of the content.
Declarative memory types include:
- "episodic" - Personal experiences
- "semantic" - Facts and general knowledge
- "autobiographical" - Personal life story
- "prospective" - Future intentions

Non-Declarative (Implicit) Memory include:
- "procedural" - Skills and procedures
- "priming" - Influence of prior exposure
- "conditioning" - Learned associations

Short-term processing memory:
"working" - Short-term processing

Based on rationale, score the importance of the memory between 0 and 1.

{format_instructions}
""".strip()


class EmotionAnalyzer(BaseAnalyzer):
    """Handles emotional content analysis."""
    
    def _get_parser(self) -> JsonOutputParser:
        return JsonOutputParser(pydantic_object=EmotionAnalysisResult)
    
    def _get_template(self) -> str:
        return """
Analyze the emotional content of the following:

Content: {content}
Context: {context}

{format_instructions}
""".strip()


class ConceptAnalyzer(BaseAnalyzer):
    """Handles concept extraction analysis."""
    
    def _get_parser(self) -> JsonOutputParser:
        return JsonOutputParser(pydantic_object=ConceptExtractionResult)
    
    def _get_template(self) -> str:
        return """
Analyze the following content and extract key concepts:

Content: {content}
Context: {context}

{format_instructions}
""".strip()


class PatternAnalyzer(BaseAnalyzer):
    """Handles pattern and insight analysis across memories."""
    
    def _get_parser(self) -> JsonOutputParser:
        return JsonOutputParser(pydantic_object=InsightsResult)
    
    def _get_template(self) -> str:
        return """
Analyze the following memories and identify key patterns, themes, and insights:

Content: {content}
Context: {context}

{format_instructions}
""".strip()
