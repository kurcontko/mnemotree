from __future__ import annotations

import asyncio
from asyncio import Lock
import json
import logging
from typing import List, Optional, Dict, Any, Tuple

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
        self._initialized = False
        
    async def initialize(self):
        """Initialize database with required indexes and constraints"""
        if self._initialized:
            return
        async with self.driver.session() as session:
            tx = await session.begin_transaction()
            try:
                # Core memory constraint
                await tx.run("""
                    CREATE CONSTRAINT memory_id IF NOT EXISTS
                    FOR (m:MemoryItem) REQUIRE m.memory_id IS UNIQUE
                """)
                
                # Entity constraints
                await tx.run("""
                    CREATE CONSTRAINT entity_text IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.text IS UNIQUE
                """)
                
                # Tag constraint
                await tx.run("""
                    CREATE CONSTRAINT tag_name IF NOT EXISTS
                    FOR (t:Tag) REQUIRE t.name IS UNIQUE
                """)
                
                # Indexes for efficient querying
                await tx.run("""
                    CREATE INDEX memory_timestamp IF NOT EXISTS
                    FOR (m:MemoryItem) ON (m.timestamp)
                """)
                
                await tx.run("""
                    CREATE INDEX memory_importance IF NOT EXISTS
                    FOR (m:MemoryItem) ON (m.importance)
                """)
                
                await tx.commit()
                logger.info("Successfully initialized Neo4j schema")
                self._initialized = True
            except Exception as e:
                await tx.rollback()
                logger.error(f"Failed to initialize Neo4j schema: {e}")
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
        
        async with memory_lock:
            async with self.driver.session() as session:
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
                    
                    # Clean and validate entities
                    valid_entities = {}
                    if memory.entities and not (len(memory.entities) == 1 and None in memory.entities):
                        valid_entities = {
                            text: etype 
                            for text, etype in memory.entities.items() 
                            if text is not None and etype is not None
                        }
                    
                    # Store the memory node with all flattened properties
                    await tx.run("""
                        CREATE (m:MemoryItem {
                            memory_id: $memory_id,
                            conversation_id: $conversation_id,
                            user_id: $user_id,
                            
                            content: $content,
                            summary: $summary,
                            author: $author,
                            memory_type: $memory_type,
                            timestamp: $timestamp,
                            
                            last_accessed: $last_accessed,
                            access_count: $access_count,
                            access_history: $access_history,
                            
                            importance: $importance,
                            decay_rate: $decay_rate,
                            confidence: $confidence,
                            fidelity: $fidelity,
                            
                            emotional_valence: $emotional_valence,
                            emotional_arousal: $emotional_arousal,
                            emotions: $emotions,
                            
                            linked_concepts: $linked_concepts,
                            previous_event_id: $previous_event_id,
                            next_event_id: $next_event_id,
                            
                            source: $source,
                            credibility: $credibility,
                            
                            embedding: $embedding,
                            context: $context,
                            
                            entities: $entities,
                            entity_mentions: $entity_mentions
                        })
                    """, {
                        'memory_id': memory.memory_id,
                        'conversation_id': memory.conversation_id,
                        'user_id': memory.user_id,
                        
                        'content': memory.content,
                        'summary': memory.summary,
                        'author': memory.author,
                        'memory_type': memory.memory_type.value,
                        'timestamp': memory.timestamp,
                        
                        'last_accessed': memory.last_accessed,
                        'access_count': memory.access_count,
                        'access_history': memory.access_history,
                        
                        'importance': memory.importance,
                        'decay_rate': memory.decay_rate,
                        'confidence': memory.confidence,
                        'fidelity': memory.fidelity,
                        
                        'emotional_valence': memory.emotional_valence,
                        'emotional_arousal': memory.emotional_arousal,
                        'emotions': memory.emotions,
                        
                        'linked_concepts': memory.linked_concepts,
                        'previous_event_id': memory.previous_event_id,
                        'next_event_id': memory.next_event_id,
                        
                        'source': memory.source,
                        'credibility': memory.credibility,
                        
                        'embedding': embedding_list,
                        'context': json.dumps(clean_context),
                        
                        'entities': json.dumps(valid_entities),
                        'entity_mentions': json.dumps(memory.entity_mentions)
                    })

                    # Create relationships
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

                    # Handle associations
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

                    # Handle conflicts
                    if memory.conflicts_with:
                        await tx.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $conflicts AS conflict_id
                            MATCH (c:MemoryItem {memory_id: conflict_id})
                            MERGE (m)-[:CONFLICTS_WITH]->(c)
                        """, {
                            'memory_id': memory.memory_id,
                            'conflicts': memory.conflicts_with
                        })
                        
                    # Create entity references while keeping the data flattened
                    if memory.entities:
                        await tx.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $entity_pairs as pair
                            MERGE (e:Entity {text: pair.text})
                            ON CREATE SET e.type = pair.type
                            MERGE (m)-[:MENTIONS_ENTITY]->(e)
                        """, {
                            'memory_id': memory.memory_id,
                            'entity_pairs': [
                                {'text': text, 'type': etype}
                                for text, etype in memory.entities.items()
                            ]
                        })
                        
                    # Handle temporal relationships
                    if memory.previous_event_id:
                        await tx.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (p:MemoryItem {memory_id: $prev_id})
                            MERGE (p)-[r:TEMPORAL_NEXT {timestamp: $timestamp}]->(m)
                        """, {
                            'memory_id': memory.memory_id,
                            'prev_id': memory.previous_event_id,
                            'timestamp': memory.timestamp
                        })
                    
                    if memory.next_event_id:
                        await tx.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (n:MemoryItem {memory_id: $next_id})
                            MERGE (m)-[r:TEMPORAL_NEXT {timestamp: $timestamp}]->(n)
                        """, {
                            'memory_id': memory.memory_id,
                            'next_id': memory.next_event_id,
                            'timestamp': memory.timestamp
                        })

                    await tx.commit()
                    logger.info(f"Successfully stored memory {memory.memory_id}")
                    
                except Exception as e:
                    await tx.rollback()
                    logger.error(f"Failed to store memory {memory.memory_id}: {e}")
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
        
    async def query_by_entities(
        self,
        entities: Dict[str, str], 
        limit: int = 10,
        depth: int = 2
    ) -> List[MemoryItem]:
        """
        Query memories by mentioned entities with deep path traversal
        
        Args:
            entities_map: Dict mapping entity text to entity type
            limit: Maximum number of memories to return
            depth: Maximum path length to traverse
        """
        async with self.driver.session() as session:
            result = await session.run("""
                UNWIND $entity_pairs as pair
                MATCH (e:Entity {text: pair.text, type: pair.type})
                MATCH path = (e)<-[:MENTIONS_ENTITY*1..2]-(m:MemoryItem)
                WITH m, 
                     collect(DISTINCT {text: e.text, type: e.type}) as matching_entities,
                     min(length(path)) as shortest_path
                ORDER BY size(matching_entities) DESC,
                         shortest_path ASC,
                         m.importance DESC
                LIMIT $limit
                RETURN m, matching_entities, shortest_path
            """, {
                'entity_pairs': [
                    {'text': text, 'type': etype}
                    for text, etype in entities.items()
                ],
                'limit': limit
            })
            
            memories = []
            async for record in result:
                try:
                    node_data = dict(record["m"])
                    context = json.loads(node_data.get('context', '{}'))
                    context.update({
                        'matching_entities': [
                            {'text': e['text'], 'type': e['type']} 
                            for e in record['matching_entities']
                        ],
                        'connection_depth': record['shortest_path']
                    })
                    node_data['context'] = context
                    
                    # Parse other JSON fields
                    for field in ['entities', 'entity_mentions']:
                        node_data[field] = json.loads(node_data.get(field, '{}'))
                    node_data['memory_type'] = MemoryType(node_data['memory_type'])
                    
                    memories.append(MemoryItem(**node_data))
                except (ValidationError, json.JSONDecodeError) as e:
                    logger.error(f"Error processing memory {node_data.get('id', 'unknown')}: {e}")
                    continue
                    
            return memories
            
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
    
    async def get_similar_memories(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 15,
        similarity_threshold: float = 0.1,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """Get similar memories with additional relationship data."""
        async with self.driver.session() as session:
            # Create temporary query node
            await session.run("""
                CREATE (:QueryEmbedding {
                    embedding: $embedding,
                    query: $query
                })
            """, {
                'embedding': query_embedding,
                'query': query
            })
            
            try:
                # Enhanced query including relationships
                result = await session.run("""
                    // First get similar memories
                    MATCH (q:QueryEmbedding), (m:MemoryItem)
                    WHERE m.embedding IS NOT NULL
                    WITH m, gds.similarity.cosine(q.embedding, m.embedding) AS score
                    WHERE score >= $threshold
                    
                    // Get tags
                    OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
                    WITH m, score, collect(DISTINCT t.name) as tags
                    
                    // Get entities
                    OPTIONAL MATCH (m)-[:MENTIONS_ENTITY]->(e:Entity)
                    WITH m, score, tags, 
                        collect(DISTINCT {text: e.text, type: e.type}) as entity_data
                    
                    // Get associations
                    OPTIONAL MATCH (m)-[:ASSOCIATED_WITH]->(a:MemoryItem)
                    WITH m, score, tags, entity_data, 
                        collect(DISTINCT a.memory_id) as associations
                    
                    // Get conflicts
                    OPTIONAL MATCH (m)-[:CONFLICTS_WITH]->(c:MemoryItem)
                    WITH m, score, tags, entity_data, associations,
                        collect(DISTINCT c.memory_id) as conflicts
                    
                    // Get temporal relationships
                    OPTIONAL MATCH (m)-[:TEMPORAL_NEXT]->(n:MemoryItem)
                    OPTIONAL MATCH (p:MemoryItem)-[:TEMPORAL_NEXT]->(m)
                    WITH m, score, tags, entity_data, associations, conflicts,
                        n.memory_id as next_id,
                        p.memory_id as prev_id
                    
                    // Return enriched results
                    RETURN 
                        m {.*} as memory,
                        score,
                        tags,
                        entity_data,
                        associations,
                        conflicts,
                        next_id,
                        prev_id
                    ORDER BY score DESC
                    LIMIT $limit
                """, {
                    'threshold': similarity_threshold,
                    'limit': top_k
                })

                memories = []
                scores = []
                
                async for record in result:
                    memory_data = record["memory"]
                    score = record["score"]
                    scores.append(score)
                    
                    try:
                        # Add relationships data
                        memory_data['tags'] = record['tags']
                        memory_data['associations'] = record['associations']
                        memory_data['conflicts_with'] = record['conflicts']
                        
                        # Add temporal relationships
                        if record['next_id']:
                            memory_data['next_event_id'] = record['next_id']
                        if record['prev_id']:
                            memory_data['previous_event_id'] = record['prev_id']
                        
                        # Parse JSON fields
                        for field in ['context', 'entities', 'entity_mentions']:
                            try:
                                memory_data[field] = json.loads(memory_data.get(field, '{}'))
                            except (json.JSONDecodeError, TypeError):
                                memory_data[field] = {}
                                
                        # Update entities from graph data
                        if record['entity_data']:
                            memory_data['entities'].update({
                                ent['text']: ent['type']
                                for ent in record['entity_data']
                            })
                        
                        # Convert memory_type
                        memory_data['memory_type'] = MemoryType(memory_data['memory_type'])
                        
                        # Create memory item
                        memory_item = MemoryItem(**memory_data)
                        memories.append(memory_item)
                        
                    except Exception as e:
                        logger.error(f"Error processing memory {memory_data.get('memory_id', 'unknown')}: {str(e)}")
                        continue

                # Log results
                if scores:
                    logger.info(f"Found {len(memories)} similar memories with scores: "
                            f"max={max(scores):.4f}, min={min(scores):.4f}, "
                            f"avg={sum(scores)/len(scores):.4f}")
                            
                    # Log relationship statistics
                    rel_stats = {
                        'with_tags': sum(1 for m in memories if m.tags),
                        'with_entities': sum(1 for m in memories if m.entities),
                        'with_associations': sum(1 for m in memories if m.associations),
                        'with_temporal': sum(1 for m in memories if m.next_event_id or m.previous_event_id)
                    }
                    logger.info(f"Relationship stats: {rel_stats}")
                else:
                    logger.warning(f"No memories found above threshold {similarity_threshold}")

                return memories
                
            finally:
                # Clean up query node
                await session.run("MATCH (q:QueryEmbedding) DELETE q")

    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: Optional[List[str]] = None,
        conflict_ids: Optional[List[str]] = None,
        previous_id: Optional[str] = None,
        next_id: Optional[str] = None
    ) -> None:
        """Update memory connections"""
        async with self.driver.session() as session:
            try:
                # 1. Update associations
                if related_ids is not None:
                    # Remove existing associations
                    await session.run("""
                        MATCH (m:MemoryItem {memory_id: $memory_id})-[r:ASSOCIATED_WITH]->()
                        DELETE r
                    """, memory_id=memory_id)
                    
                    # Add new associations
                    if related_ids:
                        await session.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $related_ids AS related_id
                            MATCH (r:MemoryItem {memory_id: related_id})
                            MERGE (m)-[:ASSOCIATED_WITH]->(r)
                        """, memory_id=memory_id, related_ids=related_ids)

                # 2. Update conflicts
                if conflict_ids is not None:
                     # Remove existing conflicts
                    await session.run("""
                        MATCH (m:MemoryItem {memory_id: $memory_id})-[r:CONFLICTS_WITH]->()
                        DELETE r
                    """, memory_id=memory_id)
                    
                    # Add new conflicts
                    if conflict_ids:
                        await session.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $conflict_ids AS conflict_id
                            MATCH (c:MemoryItem {memory_id: conflict_id})
                            MERGE (m)-[:CONFLICTS_WITH]->(c)
                        """, memory_id=memory_id, conflict_ids=conflict_ids)
                
                # 3. Update temporal relationships
                if previous_id is not None:
                    # Remove incoming temporal next to this memory (which means this is next of previous)
                     await session.run("""
                        MATCH (p:MemoryItem)-[r:TEMPORAL_NEXT]->(m:MemoryItem {memory_id: $memory_id})
                        DELETE r
                    """, memory_id=memory_id)
                     
                     if previous_id:
                        await session.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (p:MemoryItem {memory_id: $previous_id})
                            MERGE (p)-[:TEMPORAL_NEXT]->(m)
                        """, memory_id=memory_id, previous_id=previous_id)

                if next_id is not None:
                     # Remove outgoing temporal next from this memory
                    await session.run("""
                        MATCH (m:MemoryItem {memory_id: $memory_id})-[r:TEMPORAL_NEXT]->()
                        DELETE r
                    """, memory_id=memory_id)
                    
                    if next_id:
                        await session.run("""
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (n:MemoryItem {memory_id: $next_id})
                            MERGE (m)-[:TEMPORAL_NEXT]->(n)
                        """, memory_id=memory_id, next_id=next_id)

                logger.info(f"Successfully updated connections for memory {memory_id}")
            except Exception as e:
                logger.error(f"Failed to update connections for memory {memory_id}: {e}")
                raise

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Execute a complex memory query"""
        # Note: This is a simplified implementation that supports basic filtering
        async with self.driver.session() as session:
            try:
                # Build MATCH and WHERE clauses
                cypher = "MATCH (m:MemoryItem)\n"
                params = {}
                where_clauses = []
                
                # Add vector similarity calculation if vector is present
                if query.vector:
                     # Neo4j vector index query usually requires a different syntax (CALL db.index.vector.queryNodes)
                     # For simplicity/compatibility we might assume brute force or pre-filtering. 
                     # However, to properly integrate with filtering, we often filter first then sort, or vector search then filter.
                     # Here we will do valid filtering first as per cypher structure.
                     pass 
                
                if query.filters:
                    for i, filter_cond in enumerate(query.filters):
                        # Construct WHERE clause based on operator
                        param_name = f"filter_{i}"
                        field = f"m.{filter_cond.field}"
                        
                        op_map = {
                            "eq": "=", "neq": "<>", "gt": ">", "gte": ">=", 
                            "lt": "<", "lte": "<="
                        }
                        
                        if filter_cond.operator in op_map:
                            where_clauses.append(f"{field} {op_map[filter_cond.operator]} ${param_name}")
                            params[param_name] = filter_cond.value
                        elif filter_cond.operator == "contains":
                             where_clauses.append(f"${param_name} IN {field}") # Assuming list field
                             params[param_name] = filter_cond.value
                        # Add other operators as needed
                
                if where_clauses:
                    cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
                
                # Limit
                cypher += f"RETURN m LIMIT {query.limit or 10}"
                
                result = await session.run(cypher, params)
                
                memories = []
                async for record in result:
                     node_data = dict(record["m"])
                     # Basic parsing
                     try:
                        node_data['context'] = json.loads(node_data.get('context', '{}'))
                     except: 
                        node_data['context'] = {}
                     
                     node_data['memory_type'] = MemoryType(node_data['memory_type'])
                     # Note: we are missing relationships in this simple query
                     memories.append(MemoryItem(**node_data))
                
                return memories

            except Exception as e:
                logger.error(f"Failed to query memories: {e}")
                raise

    async def update_memory_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata fields for a memory item."""
        allowed_fields = {"last_accessed", "access_count", "access_history"}
        updates = {k: v for k, v in metadata.items() if k in allowed_fields}
        if not updates:
            return False

        set_clauses = []
        params: Dict[str, Any] = {"memory_id": memory_id}
        for key, value in updates.items():
            set_clauses.append(f"m.{key} = ${key}")
            params[key] = value

        cypher = f"""
            MATCH (m:MemoryItem {{memory_id: $memory_id}})
            SET {", ".join(set_clauses)}
            RETURN m
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, params)
            record = await result.single()
            return record is not None

    async def close(self):
        """Close the database connection"""
        await self.driver.close()
