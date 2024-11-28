from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json

import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    MilvusClient
)

from .base import BaseMemoryStorage
from ..core.models import MemoryItem, MemoryType, MemoryCategory


class MilvusMemoryStorage(BaseMemoryStorage):
    def __init__(self, host: str = 'localhost', port: int = '19530'):
        """Initialize Milvus connection and set up collections."""
        connections.connect(host=host, port=port)
        self.client = MilvusClient(host=host, port=port)
        self._setup_database()
    
    def _setup_database(self):
        """Set up necessary collections and indexes."""
        memory_fields = [
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="memory_category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="importance", dtype=DataType.FLOAT),
            FieldSchema(name="decay_rate", dtype=DataType.FLOAT),
            FieldSchema(name="last_accessed", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="access_count", dtype=DataType.INT64),
            FieldSchema(name="emotional_valence", dtype=DataType.FLOAT),
            FieldSchema(name="emotional_arousal", dtype=DataType.FLOAT),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="source_credibility", dtype=DataType.FLOAT),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="fidelity", dtype=DataType.FLOAT),
            # Vector field for embeddings
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            # JSON fields stored as strings
            FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="sensory_data", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            # Arrays stored as JSON strings
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="emotions", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="associations", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="linked_concepts", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="conflicts_with", dtype=DataType.VARCHAR, max_length=65535),
            # Timeline relationships
            FieldSchema(name="previous_event_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="next_event_id", dtype=DataType.VARCHAR, max_length=200),
        ]

        # Create collections if they don't exist
        if not utility.has_collection("memories"):
            memories_schema = CollectionSchema(fields=memory_fields, description="Memory storage")
            self.memories = Collection("memories", memories_schema)
            
            # Create indexes
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.memories.create_index(field_name="embedding", index_params=index_params)
            self.memories.create_index(field_name="memory_type")
            self.memories.create_index(field_name="memory_category")
            self.memories.create_index(field_name="importance")
        else:
            self.memories = Collection("memories")
            self.memories.load()

    def _prepare_memory_data(self, memory: MemoryItem) -> Dict:
        """Prepare memory data for Milvus storage."""
        data = memory.dict()
        
        # Convert enums to strings
        data['memory_type'] = memory.memory_type.value
        data['memory_category'] = memory.memory_category.value
        
        # Convert lists and dicts to JSON strings
        for field in ['context', 'sensory_data', 'metadata']:
            if data[field]:
                data[field] = json.dumps(data[field])
        
        # Convert array fields to JSON strings
        array_fields = ['tags', 'emotions', 'associations', 'linked_concepts', 'conflicts_with']
        for field in array_fields:
            if data[field]:
                data[field] = json.dumps(data[field])
            
        return data

    def store_memory(self, memory: MemoryItem) -> None:
        """Store a memory item in Milvus."""
        data = self._prepare_memory_data(memory)
        
        # Check if memory exists
        existing = self.memories.query(f"memory_id == '{memory.memory_id}'")
        
        if existing:
            # Update existing memory
            self.memories.delete(f"memory_id == '{memory.memory_id}'")
        
        # Insert memory data
        self.memories.insert([data])
        self.memories.flush()

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item."""
        results = self.memories.query(
            expr=f"memory_id == '{memory_id}'",
            output_fields=self.memories.schema.fields
        )
        
        if not results:
            return None
            
        memory_data = results[0]
        
        # Convert JSON strings back to objects
        for field in ['context', 'sensory_data', 'metadata']:
            if memory_data[field]:
                memory_data[field] = json.loads(memory_data[field])
        
        # Convert array JSON strings back to lists
        array_fields = ['tags', 'emotions', 'associations', 'linked_concepts', 'conflicts_with']
        for field in array_fields:
            if memory_data[field]:
                memory_data[field] = json.loads(memory_data[field])
            else:
                memory_data[field] = []
        
        # Convert types
        memory_data['memory_type'] = MemoryType(memory_data['memory_type'])
        memory_data['memory_category'] = MemoryCategory(memory_data['memory_category'])
        
        return MemoryItem(**memory_data)

    def get_similar_memories(self, query_embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        """Retrieve similar memories using vector similarity."""
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = self.memories.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["memory_id"]
        )
        
        memories = []
        for hits in results:
            for hit in hits:
                memory = self.get_memory(hit.entity.get('memory_id'))
                if memory:
                    memories.append(memory)
        
        return memories

    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Retrieve all memories of a specific type."""
        results = self.memories.query(
            expr=f"memory_type == '{memory_type.value}'",
            output_fields=self.memories.schema.fields
        )
        
        return [self.get_memory(result['memory_id']) for result in results]

    def update_memory_importance(self, memory_id: str, current_time: datetime) -> None:
        """Update memory importance based on decay and access."""
        memory = self.get_memory(memory_id)
        if memory:
            memory.decay_importance(current_time)
            memory.update_access()
            self.store_memory(memory)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        expr = f"memory_id == '{memory_id}'"
        result = self.memories.delete(expr)
        self.memories.flush()
        return bool(result)

    def close(self):
        """Close the Milvus connection."""
        connections.disconnect("default")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()