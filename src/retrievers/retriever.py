from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timezone
import math

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field

from ..store.base import BaseMemoryStorage
from ..memory.processor import MemoryProcessor
from ..core.models import MemoryItem, MemoryType, MemoryCategory


class MemoryRetriever(BaseRetriever):
    """A LangChain retriever that integrates with the Neo4j memory storage system."""
    
    storage: BaseMemoryStorage
    processor: MemoryProcessor
    search_type: Literal["semantic", "tag", "hybrid"] = Field(
        default="hybrid",
        description="Type of search to perform"
    )
    score_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )
    importance_weight: float = Field(
        default=0.5,
        description="Weight given to memory importance in ranking",
        ge=0.0,
        le=1.0
    )
    recency_weight: float = Field(
        default=0.3,
        description="Weight given to memory recency in ranking",
        ge=0.0,
        le=1.0
    )
    access_weight: float = Field(
        default=0.2,
        description="Weight given to memory access frequency in ranking",
        ge=0.0,
        le=1.0
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        gt=0
    )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get relevant documents based on the query."""
        # Get query embedding
        query_embedding = await self.processor.embeddings.aembed_query(query)
        
        # Get similar memories using vector similarity
        similar_memories = self.storage.get_similar_memories(
            query_embedding=query_embedding,
            top_k=self.max_results * 2  # Get more results for reranking
        )
        if not similar_memories:
            return []
        
        # Convert memories to documents with metadata and score them
        docs_with_scores = []
        current_time = datetime.now(timezone.utc)
        
        for memory in similar_memories:
            # Calculate semantic similarity score (already sorted by similarity)
            base_score = self._calculate_base_score(memory, current_time)
            
            # Create document with metadata
            doc = Document(
                page_content=memory.content,
                metadata={
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type.value,
                    "memory_category": memory.memory_category.value,
                    "importance": memory.importance,
                    "timestamp": memory.timestamp,
                    "last_accessed": memory.last_accessed,
                    "emotional_valence": memory.emotional_valence,
                    "emotional_arousal": memory.emotional_arousal,
                    "emotions": memory.emotions,
                    "tags": memory.tags,
                    "linked_concepts": memory.linked_concepts,
                    "context": memory.context,
                    "score": base_score
                }
            )
            docs_with_scores.append((doc, base_score))
            
            # Update memory access stats
            self.storage.update_memory_importance(memory.memory_id, current_time)
        
        # Sort by score and take top results
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in docs_with_scores[:self.max_results] 
                if score >= self.score_threshold]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Synchronous version of document retrieval."""
        import asyncio
        return asyncio.run(self._aget_relevant_documents(
            query, run_manager=run_manager, **kwargs
        ))

    def _calculate_base_score(
        self,
        memory: MemoryItem,
        current_time: datetime
    ) -> float:
        """Calculate the base relevance score for a memory."""
        
        # Convert timestamp strings to datetime objects
        memory_time = datetime.strptime(memory.timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
        last_accessed = datetime.strptime(memory.last_accessed, "%Y-%m-%d %H:%M:%S.%f%z")
        
        # Calculate recency score using exponential decay
        time_diff = (current_time - memory_time).total_seconds()
        decay_constant = 24 * 3600  # 24 hours
        recency_score = math.exp(-time_diff / decay_constant)
        
        # Calculate access frequency score using logarithmic scaling
        access_score = math.log(memory.access_count + 1)
        max_access = 100  # Define based on expected max access_count
        normalized_access = access_score / math.log(max_access + 1)
        
        # Optionally, calculate last accessed recency
        last_access_diff = (current_time - last_accessed).total_seconds()
        last_access_recency = math.exp(-last_access_diff / decay_constant)
        
        # Normalize importance (assuming importance is on a scale, e.g., 0-1)
        max_importance = 1.0  # Define based on your importance scale
        normalized_importance = memory.importance / max_importance
        
        # Normalize and assign weights
        importance_weight, recency_weight, access_weight = self.normalize_weights(
            self.importance_weight,
            self.recency_weight,
            self.access_weight
        )
        
        # Combine scores
        total_score = (
            importance_weight * normalized_importance +
            recency_weight * recency_score +
            access_weight * normalized_access
        )
        
        return total_score

    def normalize_weights(self, importance_weight, recency_weight, access_weight):
        total_weight = importance_weight + recency_weight + access_weight
        return (
            importance_weight / total_weight,
            recency_weight / total_weight,
            access_weight / total_weight
        )

    async def agenerate_memory(
        self,
        query: str,
        response: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """Generate and store a new memory from a query-response pair."""
        # Create memory using processor
        memory = await self.processor.create_memory_from_messages(
            prompt=query,
            response=response,
            conversation_id=conversation_id,
            additional_context=context
        )
        
        # Store memory
        self.storage.store_memory(memory)
        
        return memory
    
    def get_memories_by_type(
        self,
        memory_type: MemoryType
    ) -> List[Document]:
        """Retrieve all memories of a specific type as documents."""
        memories = self.storage.get_memories_by_type(memory_type)
        
        return [
            Document(
                page_content=memory.content,
                metadata={
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type.value,
                    "memory_category": memory.memory_category.value,
                    "importance": memory.importance,
                    "timestamp": memory.timestamp,
                    "emotional_valence": memory.emotional_valence,
                    "emotional_arousal": memory.emotional_arousal,
                    "emotions": memory.emotions,
                    "tags": memory.tags,
                    "linked_concepts": memory.linked_concepts,
                    "context": memory.context
                }
            )
            for memory in memories
        ]