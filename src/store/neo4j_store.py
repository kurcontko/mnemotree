from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from neo4j import GraphDatabase
import json

from .base import BaseMemoryStorage
from ..core.models import MemoryType, MemoryCategory, MemoryItem


class Neo4jMemoryStorage(BaseMemoryStorage):
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._setup_database()
    
    def _setup_database(self):
        """Set up necessary indexes and constraints."""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (m:MemoryItem) 
                REQUIRE m.memory_id IS UNIQUE
            """)
            
            # Create indexes for frequent lookups
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (m:MemoryItem) ON (m.memory_type)",
                "CREATE INDEX IF NOT EXISTS FOR (m:MemoryItem) ON (m.memory_category)",
                "CREATE INDEX IF NOT EXISTS FOR (m:MemoryItem) ON (m.importance)",
                "CREATE INDEX IF NOT EXISTS FOR (m:MemoryItem) ON (m.timestamp)"
            ]
            for index in indexes:
                session.run(index)

    def _prepare_memory_data(self, memory: MemoryItem) -> Dict:
        """Prepare memory data for Neo4j storage by converting all data types to Neo4j-compatible formats."""
        data = memory.model_dump()
        
        # Convert enums to strings
        data['memory_type'] = memory.memory_type.value
        data['memory_category'] = memory.memory_category.value
        
        # Convert all datetime objects to ISO format strings
        data['timestamp'] = str(data['timestamp'])
        data['last_accessed'] = str(data['last_accessed'])
        
        # Convert embedding to list of floats if it exists
        if data.get('embedding') is not None:
            data['embedding'] = [float(x) for x in data['embedding']]
        
        # Convert complex dictionaries to JSON strings
        complex_fields = ['context', 'sensory_data', 'metadata']
        for field in complex_fields:
            if data.get(field):
                # Check if the dictionary is empty
                if isinstance(data[field], dict) and not data[field]:
                    data[field] = None  # Set empty dict to None
                else:
                    try:
                        data[field] = json.dumps(data[field])
                    except (TypeError, ValueError):
                        data[field] = None
            else:
                data[field] = None
        
        # Convert lists to JSON strings if they contain non-primitive types
        list_fields = ['emotions', 'tags', 'linked_concepts']
        for field in list_fields:
            if field in data:
                # Check if the list is empty
                if isinstance(data[field], list) and not data[field]:
                    data[field] = None  # Set empty list to None
                elif isinstance(data[field], list) and all(isinstance(item, str) for item in data[field]):
                    # Leave the list as-is if it's a valid list of strings
                    continue
                else:
                    # Handle invalid list content by setting it to None
                    data[field] = None
            else:
                data[field] = None
        
        # Remove relationship fields from node properties
        # These will be handled separately as Neo4j relationships
        relationship_fields = [
            'associations',
            'linked_concepts',
            'conflicts_with',
            'previous_event_id',
            'next_event_id'
        ]
        for field in relationship_fields:
            data.pop(field, None)
        
        # Ensure all numeric values are proper primitives
        numeric_fields = ['importance', 'decay_rate', 'access_count', 
                        'emotional_valence', 'emotional_arousal',
                        'source_credibility', 'confidence', 'fidelity']
        for field in numeric_fields:
            if data.get(field) is not None:
                try:
                    data[field] = float(data[field])
                except (TypeError, ValueError):
                    data[field] = None
        
        # Remove None values to prevent Neo4j errors
        data = {k: v for k, v in data.items() if v is not None}
        
        return data

    def store_memory(self, memory: MemoryItem) -> None:
        """Store a memory item in Neo4j with all its relationships."""
        with self.driver.session() as session:
            # Prepare memory data
            memory_data = self._prepare_memory_data(memory)
            
            # Create or update the memory node using UNWIND for better performance
            create_query = """
            MERGE (m:MemoryItem {memory_id: $memory_id})
            SET m += $data
            """
            
            session.run(create_query, 
                memory_id=memory.memory_id,
                data=memory_data
            )
            
            # Handle relationships using separate transactions for better error handling
            if memory.tags:
                session.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    UNWIND $tags as tag
                    MERGE (t:Tag {name: tag})
                    MERGE (m)-[:HAS_TAG]->(t)
                """, memory_id=memory.memory_id, tags=memory.tags)
            
            if memory.associations:
                session.run("""
                    MATCH (m1:MemoryItem {memory_id: $memory_id})
                    UNWIND $associations as assoc_id
                    MATCH (m2:MemoryItem {memory_id: assoc_id})
                    MERGE (m1)-[:ASSOCIATED_WITH]->(m2)
                """, memory_id=memory.memory_id, associations=memory.associations)
            
            if memory.linked_concepts:
                session.run("""
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    UNWIND $linked_concepts as concept
                    MERGE (c:Concept {name: concept})
                    MERGE (m)-[:LINKED_TO]->(c)
                """, memory_id=memory.memory_id, linked_concepts=memory.linked_concepts)
            
            if memory.previous_event_id:
                session.run("""
                    MATCH (m1:MemoryItem {memory_id: $memory_id})
                    MATCH (m2:MemoryItem {memory_id: $prev_id})
                    MERGE (m2)-[:NEXT_EVENT]->(m1)
                """, memory_id=memory.memory_id, prev_id=memory.previous_event_id)
            
            if memory.next_event_id:
                session.run("""
                    MATCH (m1:MemoryItem {memory_id: $memory_id})
                    MATCH (m2:MemoryItem {memory_id: $next_id})
                    MERGE (m1)-[:NEXT_EVENT]->(m2)
                """, memory_id=memory.memory_id, next_id=memory.next_event_id)

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item with all its relationships."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:MemoryItem {memory_id: $memory_id})
                OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
                OPTIONAL MATCH (m)-[:ASSOCIATED_WITH]->(a:MemoryItem)
                OPTIONAL MATCH (m)-[:LINKED_TO]->(c:Concept)
                OPTIONAL MATCH (m)-[:CONFLICTS_WITH]->(conf:MemoryItem)
                OPTIONAL MATCH (prev)-[:NEXT_EVENT]->(m)
                OPTIONAL MATCH (m)-[:NEXT_EVENT]->(next)
                RETURN m,
                       collect(DISTINCT t.name) as tags,
                       collect(DISTINCT a.memory_id) as associations,
                       collect(DISTINCT c.name) as concepts,
                       collect(DISTINCT conf.memory_id) as conflicts,
                       prev.memory_id as previous_event_id,
                       next.memory_id as next_event_id
            """, memory_id=memory_id)
            
            record = result.single()
            if not record:
                return None
                
            node = record["m"]
            
            # Convert node properties back to MemoryItem
            memory_data = dict(node)
            
            # Parse JSON strings back to dictionaries
            for field in ['context', 'sensory_data', 'metadata']:
                if memory_data.get(field):
                    memory_data[field] = json.loads(memory_data[field])
            
            # Add relationship data
            memory_data.update({
                'tags': record['tags'],
                'associations': record['associations'],
                'linked_concepts': record['concepts'],
                'conflicts_with': record['conflicts'],
                'previous_event_id': record['previous_event_id'],
                'next_event_id': record['next_event_id']
            })
            
            # Convert strings back to enums
            memory_data['memory_type'] = MemoryType(memory_data['memory_type'])
            memory_data['memory_category'] = MemoryCategory(memory_data['memory_category'])
            
            return MemoryItem(**memory_data)

    def update_memory_importance(self, memory_id: str, current_time: datetime) -> None:
        """Update memory importance based on decay and access."""
        memory = self.get_memory(memory_id)
        if memory:
            memory.decay_importance(current_time)
            memory.update_access()
            self.store_memory(memory)

    def get_similar_memories(self, query_embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        """Retrieve similar memories using vector similarity."""
        with self.driver.session() as session:
            # Create temporary node with query embedding
            session.run("""
                CREATE (:QueryNode {
                    embedding: $embedding,
                    query_id: 'temp-query'
                })
            """, embedding=query_embedding)
            
            try:
                # Project graph for similarity search
                session.run("""
                    CALL gds.graph.project.cypher(
                        'similarity-graph',
                        'MATCH (n) WHERE n:MemoryItem OR n:QueryNode RETURN id(n) AS id, n.embedding AS embedding',
                        'MATCH (n)-[r:ASSOCIATED_WITH]->(m) RETURN id(n) AS source, id(m) AS target'
                    )
                """)
                # Perform k-NN search
                result = session.run("""
                    MATCH (q:QueryNode {query_id: 'temp-query'})
                    CALL gds.knn.stream('similarity-graph', {
                        nodeProperties: ['embedding'],
                        topK: $top_k
                    })
                    YIELD node1, node2, similarity
                    WHERE id(q) = node1
                    WITH node2, similarity
                    MATCH (m:MemoryItem)
                    WHERE id(m) = node2
                    RETURN m.memory_id AS memory_id, similarity
                    ORDER BY similarity DESC
                """, top_k=top_k)
                
                # Process results
                memories = []
                for record in result:
                    memory = self.get_memory(record["memory_id"])
                    if memory:
                        memories.append(memory)
                return memories
                
            finally:
                # Clean up
                session.run("MATCH (n:QueryNode {query_id: 'temp-query'}) DELETE n")
                try:
                    session.run("CALL gds.graph.drop('similarity-graph', false)")
                except Exception:
                    # Ignore errors if graph doesn't exist
                    pass

    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Retrieve all memories of a specific type."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:MemoryItem {memory_type: $type})
                RETURN m.memory_id AS memory_id
            """, type=memory_type.value)
            
            return [self.get_memory(record["memory_id"]) 
                   for record in result if record["memory_id"]]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item and all its relationships."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:MemoryItem {memory_id: $memory_id})
                DETACH DELETE m
                RETURN count(m) as deleted
            """, memory_id=memory_id)
            
            return result.single()["deleted"] > 0

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()