from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import json
import asyncio

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from ..core.models import MemoryType, MemoryCategory, MemoryItem, EmotionCategory
from .models import MemoryAnalysisResult, MemoryClassificationResult, EmotionAnalysisResult, ConceptExtractionResult


class MemoryProcessor:
    def __init__(self, model_name: str = "gpt-4o-mini", embeddings_model: str = "text-embedding-3-small") -> None:
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        
        # Initialize analysis chains
        self.memory_classifier = self._create_memory_classifier()
        self.emotion_analyzer = self._create_emotion_analyzer()
        self.concept_extractor = self._create_concept_extractor()
        
    def _create_memory_classifier(self) -> Runnable:
        """Create a chain for classifying memory type and importance."""
        parser = JsonOutputParser(pydantic_object=MemoryClassificationResult)

        template = """
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

        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser

    def _create_emotion_analyzer(self) -> Runnable:
        """Create a chain for emotional analysis."""
        parser = JsonOutputParser(pydantic_object=EmotionAnalysisResult)

        template =  """
Analyze the emotional content of the following:

Content: {content}
Context: {context}

{format_instructions}
""".strip()

        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser


    def _create_concept_extractor(self) -> Runnable:
        """Create a chain for extracting concepts and tags."""
        parser = JsonOutputParser(pydantic_object=ConceptExtractionResult)

        template =  """
Analyze the following content and extract key concepts:

Content: {content}
Context: {context}

{format_instructions}
""".strip()

        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser

    async def create_memory_from_messages(
        self,
        prompt: str,
        response: str,
        conversation_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """Create a memory item from an LLM exchange."""
        
        # Combine prompt and response for analysis
        content = f"User: {prompt}\nAssistant: {response}"
        context = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(additional_context or {})
        }
        
        # Run parallel analysis
        results, embeddings = await asyncio.gather(
            self._analyze_content(content, context),
            self._get_embedding(content)
        )
        
        # Create memory item
        memory = MemoryItem(
            content=content,
            memory_category=results.memory_category,
            memory_type=results.memory_type,
            importance=results.importance,
            emotional_valence=results.emotional_valence,
            emotional_arousal=results.emotional_arousal,
            emotions=results.emotions,
            tags=results.tags,
            linked_concepts=results.linked_concepts,
            context=results.context_summary,
            embedding=embeddings
        )
        
        return memory

    async def _analyze_content(
        self, 
        content: str, 
        context: Dict[str, Any]
    ) -> MemoryAnalysisResult:
        """Perform comprehensive content analysis."""
        
        # Run all analyses in parallel
        classification, emotional, concepts = await asyncio.gather(
            self.memory_classifier.ainvoke({
                "content": content,
                "context": json.dumps(context)
            }),
            self.emotion_analyzer.ainvoke({
                "content": content,
                "context": json.dumps(context)
            }),
            self.concept_extractor.ainvoke({
                "content": content,
                "context": json.dumps(context)
            })
        )
        
        # # Parse JSON responses
        classification = MemoryClassificationResult(**classification)
        emotional = EmotionAnalysisResult(**emotional)
        concepts = ConceptExtractionResult(**concepts)

        return MemoryAnalysisResult(
            memory_type=MemoryType(classification.memory_type),
            memory_category=MemoryCategory(classification.memory_category),
            importance=classification.importance,
            emotions=emotional.emotions,
            emotional_valence=emotional.emotional_valence,
            emotional_arousal=emotional.emotional_arousal,
            tags=concepts.tags,
            linked_concepts=concepts.linked_concepts,
            context_summary=concepts.context_summary
        )

    async def _get_embedding(self, content: str) -> List[float]:
        """Get embedding for content."""
        return await self.embeddings.aembed_query(content)