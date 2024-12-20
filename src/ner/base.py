from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from enum import Enum
import spacy
from pydantic import BaseModel
from langchain_core.language_models.base import BaseLanguageModel


class NERResult(BaseModel):
    """Structured output from NER processing."""
    entities: Dict[str, str]  # Entity text -> Entity type
    mentions: Dict[str, List[str]]  # Entity text -> List of context snippets
    confidence: Optional[Dict[str, float]] = None  # Entity text -> Confidence score


class BaseNER(ABC):
    """Abstract base class for NER implementations."""
    
    @abstractmethod
    async def extract_entities(self, text: str) -> NERResult:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            NERResult containing extracted entities and their mentions
        """
        pass
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for an entity mention."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]