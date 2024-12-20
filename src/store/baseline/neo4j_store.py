import asyncio
from asyncio import Lock
import json
import logging
from typing import List, Optional, Dict, Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError
import numpy as np
from pydantic import BaseModel, ValidationError

from ...core.models import MemoryItem, MemoryType, EmotionCategory
from ...core.query import MemoryQuery
from ..base import BaseMemoryStore


logger = logging.getLogger(__name__)


class BaselineNeo4jMemoryStore(BaseMemoryStore):
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._memory_locks = {}
        self._global_lock = Lock()
        
    async def initialize(self):
        """Initialize database with required indexes and constraints"""
        async with self.driver.session() as session:
            # Begin transaction properly with await
            tx = await session.begin_transaction()
            try:
                await tx.run("""
                    CREATE CONSTRAINT memory_id IF NOT EXISTS
                    FOR (m:MemoryItem) REQUIRE m.memory_id IS UNIQUE
                """)
                
                # Use a RANGE index for embedding (works for arrays/large properties)
                await tx.run("""
                    DROP INDEX memory_embedding IF EXISTS
                """)
                await tx.commit()
            except Exception as e:
                await tx.rollback()
                raise
    
    async def _get_memory_lock(self, memory_id: str) -> Lock:
        """Get or create a lock for a specific memory"""
        async with self._global_lock:
            if memory_id not in self._memory_locks:
                self._memory_locks[memory_id] = Lock()
            return self._memory_locks[memory_id]

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a single memory with transaction safety"""
        memory_lock = await self._get_memory_lock(memory.memory_id)
        
        async with memory_lock:  # Prevent concurrent modifications
            async with self.driver.session() as session:
                # Begin transaction properly with await
                tx = await session.begin_transaction()
                try:
                    # Prepare data
                    embedding_list = (memory.embedding.tolist() 
                                   if isinstance(memory.embedding, np.ndarray) 
                                   else memory.embedding)
                    
                    # Store the memory node
                    await tx.run("""
                        CREATE (m:MemoryItem {
                            memory_id: $memory_id,
                            content: $content,
                            memory_type: $memory_type,
                            timestamp: $timestamp,
                            embedding: $embedding,
                            importance: $importance
                        })
                    """, {
                        'memory_id': memory.memory_id,
                        'content': memory.content,
                        'memory_type': memory.memory_type.value,
                        'timestamp': memory.timestamp,
                        'embedding': embedding_list,
                        'importance': memory.importance,
                    })

                    await tx.commit()
                    logger.info(f"Successfully stored memory {memory.memory_id}")
                    
                except Exception as e:
                    await tx.rollback()
                    logger.error(f"Failed to store memory {memory.memory_id}: {e}")
                    raise
                
    async def get_entity_contexts(self, entities: List[str]) -> List[MemoryItem]:
        # Given a list of entity texts, find related memory items
        return []
            
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory with transaction safety"""
        return None
            
    async def delete_memory(
        self,
        memory_id: str,
        *,
        cascade: bool = False
    ) -> bool:
        """
        Delete a memory item and optionally its relationships.

        Args:
            memory_id (str): The unique identifier of the memory.
            cascade (bool, optional): If True, delete related connections. Defaults to False.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        return False

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Get similar memories using cosine similarity"""
        async with self.driver.session() as session:
            # Create a temporary node with query embedding
            await session.run("""
                CREATE (:QueryEmbedding {embedding: $embedding})
            """, {'embedding': query_embedding})
            
            # Calculate cosine similarity and get top results
            result = await session.run("""
                MATCH (q:QueryEmbedding), (m:MemoryItem)
                WITH m, gds.similarity.cosine(q.embedding, m.embedding) AS similarity
                ORDER BY similarity DESC
                LIMIT $limit
                RETURN m {
                    .memory_id, .content, .memory_type, .embedding, .importance
                } AS memory,
                similarity
            """, {'limit': top_k})
            
            # Clean up query node
            await session.run("MATCH (q:QueryEmbedding) DELETE q")
            
            # Convert results to MemoryItems
            memories = []
            async for record in result:
                memory_data = record['memory']
                try:
                    memory_type = MemoryType(memory_data['memory_type'])
                    memory_item = MemoryItem(
                        memory_id=memory_data['memory_id'],
                        content=memory_data['content'],
                        memory_type=memory_type,
                        embedding=memory_data['embedding'],
                        importance=memory_data['importance'],
                    )
                except ValidationError as e:
                    logger.error("Validation error when creating MemoryItem:")
                    for error in e.errors():
                        logger.error(
                            f"Field: {' -> '.join(error['loc'])}, "
                            f"Error: {error['msg']}, "
                            f"Input value: {error.get('input', 'N/A')}, "
                            f"Expected type: {error.get('type', 'N/A')}"
                        )
                    raise
                memories.append(memory_item)
            
            return memories

    async def update_connections(
        self,
        memory_id: str,
        related_ids: Optional[List[str]] = None,
        conflict_ids: Optional[List[str]] = None,
        previous_id: Optional[str] = None,
        next_id: Optional[str] = None
    ) -> None:
        """Update memory connections"""
        raise NotImplementedError("Method not implemented")

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Execute a complex memory query"""
        raise NotImplementedError("Method not implemented")

    async def close(self):
        """Close the database connection"""
        await self.driver.close()