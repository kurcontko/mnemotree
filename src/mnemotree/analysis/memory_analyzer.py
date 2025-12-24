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
from .analyzers import (
    MemoryClassifierAnalyzer, 
    EmotionAnalyzer, 
    ConceptAnalyzer, 
    PatternAnalyzer
)


class MemoryAnalyzer:
    """Orchestrates different types of memory analysis."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings
    ):
        self.llm = llm
        self.embeddings = embeddings
        self._init_analyzers()
    
    def _init_analyzers(self) -> None:
        """Initialize individual analyzers."""
        self.memory_classifier = MemoryClassifierAnalyzer(self.llm)
        self.emotion_analyzer = EmotionAnalyzer(self.llm)
        self.concept_analyzer = ConceptAnalyzer(self.llm)
        self.pattern_analyzer = PatternAnalyzer(self.llm)
    
    async def analyze(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MemoryAnalysisResult:
        """Perform comprehensive content analysis."""
        context_str = json.dumps(context or {})
        
        # Run analyses in parallel
        classification, emotional, concepts = await asyncio.gather(
            self.memory_classifier.analyze(content, context_str),
            self.emotion_analyzer.analyze(content, context_str),
            self.concept_analyzer.analyze(content, context_str)
        )
        
        # Parse results
        classification = MemoryClassificationResult(**classification)
        emotional = EmotionAnalysisResult(**emotional)
        concepts = ConceptExtractionResult(**concepts)
        
        return MemoryAnalysisResult(
            memory_type=MemoryType(classification.memory_type),
            importance=classification.importance,
            emotions=emotional.emotions,
            emotional_valence=emotional.emotional_valence,
            emotional_arousal=emotional.emotional_arousal,
            tags=concepts.tags,
            linked_concepts=concepts.linked_concepts,
            context_summary=concepts.context_summary
        )
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        return await self.embeddings.aembed_query(text)
    
    async def analyze_patterns(self, content: str, context: str = "") -> Dict[str, Any]:
        """Analyzes multiple memories for patterns and insights."""
        return await self.pattern_analyzer.analyze(content, context)
    
    async def analyze_emotions(self, content: str, context: str = "") -> Dict[str, Any]:
        """Analyzes emotional content."""
        return await self.emotion_analyzer.analyze(content, context)
    
    async def analyze_concepts(self, content: str, context: str = "") -> Dict[str, Any]:
        """Extracts concepts from content."""
        return await self.concept_analyzer.analyze(content, context)
    
    async def analyze_memory(self, content: str, context: str = "") -> Dict[str, Any]:
        """Analyzes memory content."""
        return await self.memory_classifier.analyze(content, context)
