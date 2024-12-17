import asyncio
from asyncio import Lock
import json
import logging
from typing import List, Optional, Dict, Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError
import numpy as np
from pydantic import BaseModel, ValidationError

from ..core.models import MemoryItem, MemoryType, EmotionCategory
from ..core.query import MemoryQuery
from .base import BaseMemoryStore


logger = logging.getLogger(__name__)


class Neo4jMemoryStore(BaseMemoryStore):
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
                    
                    # Clean dictionaries for Neo4j
                    clean_context = {
                        k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
                        for k, v in (memory.context or {}).items()
                    }
                    
                    # Store the memory node
                    await tx.run("""
                        CREATE (m:MemoryItem {
                            memory_id: $memory_id,
                            content: $content,
                            memory_type: $memory_type,
                            timestamp: $timestamp,
                            importance: $importance,
                            embedding: $embedding,
                            context: $context,
                            
                            tags: $tags,
                            emotions: $emotions,
                            
                            access_count: $access_count,
                            last_accessed: $last_accessed,
                            
                            confidence: $confidence,
                            source: $source
                        })
                    """, {
                        'memory_id': memory.memory_id,
                        'content': memory.content,
                        'memory_type': memory.memory_type.value,
                        'timestamp': memory.timestamp,
                        'importance': memory.importance,
                        'embedding': embedding_list,
                        'context': json.dumps(clean_context),
                        
                        'tags': memory.tags,
                        'emotions': [e.value if isinstance(e, EmotionCategory) else e 
                                   for e in memory.emotions],
                        
                        'access_count': memory.access_count,
                        'last_accessed': memory.last_accessed,
                        
                        'confidence': memory.confidence,
                        'source': memory.source
                    })

                    # Create relationships in the same transaction
                    if memory.tags:
                        await tx.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $tags AS tag
                            MERGE (t:Tag {name: tag})
                            MERGE (m)-[:HAS_TAG]->(t)
                        """, {
                            'memory_id': memory.memory_id,
                            'tags': memory.tags
                        })

                    if memory.associations:
                        await tx.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $associations AS assoc_id
                            MATCH (a:MemoryItem {memory_id: assoc_id})
                            MERGE (m)-[:ASSOCIATED_WITH]->(a)
                        """, {
                            'memory_id': memory.memory_id,
                            'associations': memory.associations
                        })

                    await tx.commit()
                    logger.info(f"Successfully stored memory {memory.memory_id}")
                    
                except Exception as e:
                    await tx.rollback()
                    logger.error(f"Failed to store memory {memory.memory_id}: {e}")
                    raise
            
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory with transaction safety"""
        async with self.driver.session() as session:
            # Begin transaction properly with await
            tx = await session.begin_transaction()
            try:
                result = await tx.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
                    OPTIONAL MATCH (m)-[:ASSOCIATED_WITH]->(a:MemoryItem)
                    RETURN m,
                           collect(DISTINCT t.name) as tags,
                           collect(DISTINCT a.memory_id) as associations
                """, memory_id=memory_id)

                record = await result.single()
                if not record:
                    await tx.commit()
                    return None

                # Get base memory properties
                node_data = dict(record["m"])
                
                # Parse JSON context
                try:
                    node_data['context'] = json.loads(node_data.get('context', '{}'))
                except json.JSONDecodeError:
                    node_data['context'] = {}

                # Add relationships
                node_data['tags'] = record['tags']
                node_data['associations'] = record['associations']

                # Convert memory_type back to enum
                node_data['memory_type'] = MemoryType(node_data['memory_type'])

                await tx.commit()
                return MemoryItem(**node_data)

            except Exception as e:
                await tx.rollback()
                logger.error(f"Failed to retrieve memory {memory_id}: {e}")
                raise
            
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
        async with self.driver.session() as session:
            try:
                if cascade:
                    # DETACH DELETE removes the node and all its relationships
                    result = await session.run("""
                        MATCH (m:MemoryItem {memory_id: $memory_id})
                        DETACH DELETE m
                        RETURN COUNT(m) AS deleted
                    """, memory_id=memory_id)
                else:
                    # Only delete the node if it has no relationships
                    result = await session.run("""
                        MATCH (m:MemoryItem {memory_id: $memory_id})
                        WHERE NOT (m)--()
                        DELETE m
                        RETURN COUNT(m) AS deleted
                    """, memory_id=memory_id)

                deleted = await result.single()
                return deleted["deleted"] > 0
            except Neo4jError as e:
                print(f"Error deleting memory {memory_id}: {e}")
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
                    .memory_id, .content, .memory_type, .memory_category,
                    .importance, .tags, .context, .embedding
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
                        importance=memory_data['importance'],
                        tags=memory_data['tags'],
                        context=memory_data['context'],
                        #embedding=np.array(memory_data['embedding'])
                        embedding=memory_data['embedding']
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
        async with self.driver.session() as session:
            # Create RELATED_TO relationships
            if related_ids:
                await session.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    UNWIND $related_ids AS related_id
                    MATCH (r:MemoryItem {memory_id: related_id})
                    MERGE (m)-[:RELATED_TO]->(r)
                """, {
                    'memory_id': memory_id,
                    'related_ids': related_ids
                })
            
            # Create CONFLICTS_WITH relationships
            if conflict_ids:
                await session.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    UNWIND $conflict_ids AS conflict_id
                    MATCH (c:MemoryItem {memory_id: conflict_id})
                    MERGE (m)-[:CONFLICTS_WITH]->(c)
                """, {
                    'memory_id': memory_id,
                    'conflict_ids': conflict_ids
                })
            
            # Create NEXT/PREVIOUS relationships
            if previous_id:
                await session.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    MATCH (p:MemoryItem {memory_id: $previous_id})
                    MERGE (p)-[:NEXT]->(m)
                """, {
                    'memory_id': memory_id,
                    'previous_id': previous_id
                })
            
            if next_id:
                await session.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    MATCH (n:MemoryItem {memory_id: $next_id})
                    MERGE (m)-[:NEXT]->(n)
                """, {
                    'memory_id': memory_id,
                    'next_id': next_id
                })

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Execute a complex memory query"""
        async with self.driver.session() as session:
            # Build base query
            cypher_query = "MATCH (m:MemoryItem) "
            params = {}
            
            # Add vector similarity if provided
            if query.vector is not None:
                params['query_vector'] = query.vector.tolist()
                cypher_query += """
                    WITH m, gds.similarity.cosine(m.embedding, $query_vector) AS similarity
                    WHERE similarity >= 0.7
                """
            
            # Add filters
            if query.filters:
                for i, filter in enumerate(query.filters):
                    param_name = f'filter_value_{i}'
                    cypher_query += f"""
                        {'AND' if i > 0 or query.vector is not None else 'WHERE'} 
                        m.{filter.field} {filter.operator.value} ${param_name}
                    """
                    params[param_name] = filter.value
            
            # Add relationships
            if query.relationships:
                for i, rel in enumerate(query.relationships):
                    cypher_query += f"""
                        {'AND' if i > 0 or query.filters else 'WHERE'}
                        EXISTS ((m)-[:{rel.type}]->(:MemoryItem))
                    """
            
            # Add return statement
            cypher_query += """
                RETURN m {
                    .memory_id, .content, .memory_type, .memory_category,
                    .importance, .tags, .context, .embedding
                } AS memory
            """
            
            # Add limit if provided
            if query.limit:
                cypher_query += f" LIMIT {query.limit}"
            
            # Execute query
            result = await session.run(cypher_query, params)
            
            # Convert results
            memories = []
            async for record in result:
                memory_data = record['memory']
                memories.append(MemoryItem(
                    memory_id=memory_data['memory_id'],
                    content=memory_data['content'],
                    memory_type=MemoryType(memory_data['memory_type']),
                    importance=memory_data['importance'],
                    tags=memory_data['tags'],
                    context=memory_data['context'],
                    #embedding=np.array(memory_data['embedding'])
                    embedding=memory_data['embedding']
                ))
            
            return memories

    async def close(self):
        """Close the database connection"""
        await self.driver.close()