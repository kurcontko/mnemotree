from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Awaitable, Literal
from datetime import datetime, timezone
import asyncio
from functools import lru_cache
from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings.embeddings import Embeddings

from .models import MemoryType, MemoryItem, EmotionCategory
from .scoring import MemoryScoring
from ..store.base import BaseMemoryStore
from ..analysis.memory_analyzer import MemoryAnalyzer
from ..analysis.models import MemoryAnalysisResult, InsightsResult
from ..analysis.clustering import MemoryClusterer, ClusteringResult
from ..analysis.summarizer import Summarizer
from .query import MemoryQuery, MemoryQueryBuilder, FilterOperator



class MemoryCore:
    """Core memory management system."""
    
    def __init__(
        self,
        store: BaseMemoryStore,
        llm: Optional[BaseLanguageModel] = None,
        embeddings: Optional[Embeddings] = None,
        default_importance: float = 0.5,
        pre_remember_hooks: Optional[List[Callable[[MemoryItem], Awaitable[MemoryItem]]]] = None,
        memory_scoring: Optional[MemoryScoring] = None
    ):
        """
        Initializes the MemoryCore.

        Args:
            storage: The underlying storage for memory items.
            llm: Optional Language model for analysis if analyzer is not set. Defaults to ChatOpenAI("gpt-4o-mini") if not provided.
            embeddings: Optional embeddings model for analysis if analyzer is not set. Defaults to OpenAIEmbeddings("text-embedding-3-small") if not provided.
            analyzer: Optional MemoryAnalyzer. If not provided, a default analyzer is created using llm and embeddings.
            default_importance: The default importance value for memories.
            pre_remember_hooks: An optional list of asynchronous functions to run on memory items before storage.
        """
        self.store = store
        
        # Setup analyzer: if not provided, create it using the llm and embeddings if provided, or their defaults
        if not llm:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        if not embeddings:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
        self.analyzer = MemoryAnalyzer(
            llm=llm, 
            embeddings=embeddings
        )
        self.summarizer = Summarizer(llm=llm)
        self.embedder = embeddings
        
        if not memory_scoring:
            memory_scoring = MemoryScoring()
            
        self.clusterer = MemoryClusterer(self.summarizer)
        
        self.memory_scoring = memory_scoring
        self.default_importance = default_importance
        self.pre_remember_hooks = pre_remember_hooks or []
    
    async def remember(
        self,
        content: str,
        *,
        memory_type: Optional[MemoryType] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        analyze: bool = True,
        summarize: bool = True,
        references: Optional[List[str]] = None,
        skip_store: bool = False
    ) -> MemoryItem:
        """
        Store a new memory with optional analysis - optimized for concurrent operations.

        Args:
            content: The memory content to store
            memory_type: Optional explicit memory type
            importance: Optional explicit importance score
            tags: Optional manual tags
            context: Optional contextual information
            analyze: Whether to perform content analysis
            summarize: Whether to generate a summary
            references: Optional list of related memory IDs
            skip_store: If True, store will not persist the data

        Returns:
            The created MemoryItem
        """
        # Create tasks for concurrent execution
        tasks = []
        
        # Always get embedding as it's required
        embedding_task = asyncio.create_task(self.get_embedding(content))
        
        if summarize:
            summary_task = asyncio.create_task(self.summarizer.summarize(content))
            tasks.append(summary_task)
        
        if analyze:
            analysis_task = asyncio.create_task(self.analyzer.analyze(content, context))
            tasks.append(analysis_task)
        
        # Wait for all analysis tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks)
            
            # Assign results based on which tasks were created
            current_idx = 0
            if summarize:
                summary = results[current_idx]
                current_idx += 1
            else:
                summary = None
                
            if analyze:
                analysis: MemoryAnalysisResult = results[current_idx]
                memory_type = memory_type or analysis.memory_type
                importance = importance or analysis.importance
                all_tags = set(tags or []) | set(analysis.tags)
            else:
                analysis = None
                memory_type = memory_type or MemoryType.SEMANTIC
                importance = importance or self.default_importance
                all_tags = set(tags or [])
        else:
            summary = None
            analysis = None
            memory_type = memory_type or MemoryType.SEMANTIC
            importance = importance or self.default_importance
            all_tags = set(tags or [])
        
        # Wait for embedding which we always need
        embedding = await embedding_task
        
        # Create initial memory item using Pydantic model
        memory_data = {
            "memory_id": str(uuid4()),
            "content": content,
            "summary": summary,
            "memory_type": memory_type,
            "importance": importance,
            "tags": list(all_tags),
            "context": context or {},
            "embedding": embedding,
            "emotional_context": analysis.emotions if analysis else None,
            "linked_concepts": analysis.linked_concepts if analysis else None
        }
        
        memory = MemoryItem(**memory_data)
        
        # Process pre-remember hooks concurrently if there are multiple
        if self.pre_remember_hooks:
            if len(self.pre_remember_hooks) == 1:
                memory = await self.pre_remember_hooks[0](memory)
            else:
                hook_results = await asyncio.gather(
                    *(hook(memory) for hook in self.pre_remember_hooks)
                )
                # Apply hook results sequentially to maintain consistency
                for hook_result in hook_results:
                    # Convert to dict, update with new values, and create new model
                    current_data = memory.model_dump()
                    hook_data = hook_result.model_dump()
                    current_data.update({
                        k: v for k, v in hook_data.items() 
                        if v is not None
                    })
                    memory = MemoryItem(**current_data)

        # Handle storage and references concurrently
        if not skip_store:
            store_tasks = [self.store.store_memory(memory)]
            if references:
                store_tasks.append(self.connect(memory.memory_id, related_to=references))
            await asyncio.gather(*store_tasks)

        return memory
    
    async def recall(
        self,
        query: Union[str, MemoryQuery, MemoryQueryBuilder],
        *,
        limit: Optional[int] = None
    ) -> List[MemoryItem]:
        """
        Retrieve memories based on a query string, MemoryQuery or a MemoryQueryBuilder.

        Args:
            query: The query string, MemoryQuery object or MemoryQueryBuilder
            limit: Optional max results to return

        Returns:
            List of matching MemoryItems
        """
        if isinstance(query, str):
            # Simple content query
            query_embedding = await self.get_embedding(query)
            memories = await self.store.get_similar_memories(
                query=query,
                query_embedding=query_embedding,
                top_k=limit or 10
            )
        elif isinstance(query, MemoryQuery):
            # Complex query
            if query.vector is None and not query.filters and not query.relationships:
              raise ValueError("Invalid MemoryQuery, vector, filters or relationships needs to be defined")
            memories = await self.store.query_memories(query)
        elif isinstance(query, MemoryQueryBuilder):
            # Complex query
            memories = await self.store.query_memories(await query.build())
        else:
            raise ValueError("Invalid query type")
        
        memories = self.memory_scoring.filter_memories_by_score(memories)
        return memories
    
    async def reflect(
        self,
        query_builder: Optional[MemoryQueryBuilder] = None,
        min_importance: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze patterns and insights across memories.

        Args:
            query_builder: Optional MemoryQueryBuilder for filtering memories.
            min_importance: Minimum importance threshold

        Returns:
            Analysis results including patterns and insights
        """
        if query_builder is None:
            query_builder = MemoryQueryBuilder().importance_range(min_value=min_importance)
        else:
           query_builder.importance_range(min_value=min_importance)

        memories = await self.store.query_memories(await query_builder.build())
        
        if not memories:
            return {
                "patterns": [],
                "insights": [],
                "summary": "No memories found for analysis"
            }
        
        # Analyze patterns and connections
        memory_texts = [
            f"Memory {i+1}: {m.content} (Type: {m.memory_type.value}, "
            f"Importance: {m.importance:.2f})"
            for i, m in enumerate(memories)
        ]
        
        analysis = await self.analyzer.analyze_patterns("\n".join(memory_texts))
        return analysis
    
    async def connect(
        self,
        memory_id: str,
        *,
        related_to: Optional[List[str]] = None,
        conflicts_with: Optional[List[str]] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None
    ) -> None:
        """
        Create connections between memories.
        
        Args:
            memory_id: Source memory ID
            related_to: IDs of related memories
            conflicts_with: IDs of conflicting memories
            previous: ID of previous memory in sequence
            next: ID of next memory in sequence
        """
        await self.store.update_connections(
            memory_id=memory_id,
            related_ids=related_to,
            conflict_ids=conflicts_with,
            previous_id=previous,
            next_id=next
        )
    
    async def forget(
        self,
        memory_id: str,
        *,
        cascade: bool = False
    ) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
            cascade: Whether to delete connected memories
            
        Returns:
            Success status
        """
        return await self.store.delete_memory(memory_id, cascade=cascade)
    
    async def summarize(
        self,
        memories: List[MemoryItem],
        format: str = "text"
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a summary of multiple memories.
        
        Args:
            memories: List of memories to summarize
            format: Output format ("text" or "structured")
            
        Returns:
            Summary in requested format
        """
        if not memories:
            return "" if format == "text" else {}
        
        memory_texts = [f"- {m.content}" for m in memories]
        return await self.summarizer.summarize(
            "\n".join(memory_texts),
            format=format
        )
    
    async def batch_remember(
        self,
        contents: List[str],
        *,
        analyze: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        Store multiple memories efficiently.
        
        Args:
            contents: List of memory contents
            analyze: Whether to analyze contents
            context: Optional shared context
            
        Returns:
            List of created memories
        """
        tasks = [
            self.remember(
                content,
                analyze=analyze,
                context=context
            )
            for content in contents
        ]
        return await asyncio.gather(*tasks)
    
    async def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search memories with optional filtering.
        
        Args:
            query: Search query
            filters: Optional filters to apply
            limit: Max results to return
            
        Returns:
            List of (memory, relevance_score) tuples
        """
        query_embedding = await self.get_embedding(query)
        
        return await self.store.query_memories(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit
        )
    
    async def cluster(
        self,
        query: Optional[Union[str, MemoryQuery, MemoryQueryBuilder]] = None,
        method: Literal["vector_dbscan", "vector_kmeans", "graph_community"] = "vector_dbscan",
        **clustering_params
    ) -> Tuple[List[MemoryItem], ClusteringResult]:
        """
        Cluster memories based on similarity.
        
        Args:
            query: Optional query to filter memories before clustering
            method: Clustering method to use
            **clustering_params: Additional parameters for clustering
            
        Returns:
            Tuple of (memories, clustering_results)
        """
        # Get memories to cluster
        if query is not None:
            memories = await self.recall(query)
        else:
            # TODO: Add support for all memories
            raise ValueError("Query is required for clustering")
            
        # Perform clustering
        results = await self.clusterer.cluster_memories(
            memories,
            method=method,
            **clustering_params
        )
        
        return memories, results
        
    def decay_and_reinforce(self):
        """
        Decay and reinforce memory importance scores.
        
        This method should be called periodically to update memory importance scores.
        """
        raise NotImplementedError("Method not implemented")
    
    async def get_embedding(self, text: str) -> Any:
        """Get the embedding for a given text."""
        return await self.embedder.aembed_query(text)