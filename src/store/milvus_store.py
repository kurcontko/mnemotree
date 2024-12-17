import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)

from ..core.models import MemoryItem, MemoryType
from ..core.query import MemoryQuery
from .base import BaseMemoryStore


logger = logging.getLogger(__name__)


class MilvusMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        uri: str = "localhost:19530",
        user: str = "",
        password: str = "",
        collection_name: str = "memories",
        dim: int = 1536,  # Default dimension for embeddings
        consistency_level: str = "Strong"
    ):
        """
        Initialize Milvus storage connection
        
        Args:
            uri: Milvus server URI
            user: Username for authentication
            password: Password for authentication
            collection_name: Name of the collection to use
            dim: Dimension of embedding vectors
            consistency_level: Milvus consistency level (Strong/Session/Bounded/Eventually)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dim = dim
        self.consistency_level = consistency_level
        self.collection: Optional[Collection] = None

    async def initialize(self):
        """Initialize Milvus connection and create collection if needed"""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=self.uri,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connected to Milvus at {self.uri}")

            # Define collection schema
            fields = [
                FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="importance", dtype=DataType.FLOAT),
                FieldSchema(name="confidence", dtype=DataType.FLOAT),
                FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="emotions", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Memory storage collection"
            )

            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level
                )

                # Create index for vector similarity search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
            else:
                self.collection = Collection(self.collection_name)
                self.collection.load()

            logger.info(f"Successfully initialized Milvus collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            raise

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a single memory"""
        try:
            # Prepare data for insertion
            data = {
                "memory_id": [memory.memory_id],
                "content": [memory.content],
                "memory_type": [memory.memory_type.value],
                "timestamp": [memory.timestamp],
                "importance": [float(memory.importance)],
                "confidence": [float(memory.confidence)],
                "tags": [json.dumps(memory.tags)],
                "emotions": [json.dumps([str(e) for e in memory.emotions])],
                "source": [memory.source if memory.source else ""],
                "context": [json.dumps(memory.context) if memory.context else "{}"],
                "embedding": [memory.embedding]
            }

            # Insert into Milvus
            self.collection.insert(data)
            self.collection.flush()
            logger.info(f"Successfully stored memory {memory.memory_id}")

        except Exception as e:
            logger.error(f"Failed to store memory {memory.memory_id}: {e}")
            raise

    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID"""
        try:
            # Query by memory_id
            self.collection.load()
            results = self.collection.query(
                expr=f'memory_id == "{memory_id}"',
                output_fields=['*']
            )

            if not results:
                return None

            # Convert result to MemoryItem
            result = results[0]
            memory_data = {
                "memory_id": result["memory_id"],
                "content": result["content"],
                "memory_type": MemoryType(result["memory_type"]),
                "timestamp": result["timestamp"],
                "importance": float(result["importance"]),
                "confidence": float(result["confidence"]),
                "tags": json.loads(result["tags"]),
                "emotions": json.loads(result["emotions"]),
                "source": result["source"] if result["source"] else None,
                "context": json.loads(result["context"]),
                "embedding": result["embedding"]
            }

            return MemoryItem(**memory_data)

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        try:
            expr = f'memory_id == "{memory_id}"'
            self.collection.delete(expr)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def get_similar_memories(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Get similar memories using vector similarity"""
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            self.collection.load()
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=['*']
            )

            memories = []
            for hits in results:
                for hit in hits:
                    memory_data = {
                        "memory_id": hit.entity.get("memory_id"),
                        "content": hit.entity.get("content"),
                        "memory_type": MemoryType(hit.entity.get("memory_type")),
                        "timestamp": hit.entity.get("timestamp"),
                        "importance": float(hit.entity.get("importance")),
                        "confidence": float(hit.entity.get("confidence")),
                        "tags": json.loads(hit.entity.get("tags")),
                        "emotions": json.loads(hit.entity.get("emotions")),
                        "source": hit.entity.get("source") if hit.entity.get("source") else None,
                        "context": json.loads(hit.entity.get("context")),
                        "embedding": hit.entity.get("embedding")
                    }
                    memories.append(MemoryItem(**memory_data))

            return memories

        except Exception as e:
            logger.error(f"Failed to get similar memories: {e}")
            raise

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Execute a complex memory query"""
        try:
            # Build Milvus expression for filtering
            expressions = []
            if query.filters:
                for filter in query.filters:
                    if isinstance(filter.value, str):
                        expressions.append(f'{filter.field} == "{filter.value}"')
                    else:
                        expressions.append(f'{filter.field} {filter.operator.value} {filter.value}')

            expr = " && ".join(expressions) if expressions else ""

            # If vector similarity is requested
            if query.vector is not None:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10},
                }

                results = self.collection.search(
                    data=[query.vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=query.limit or 10,
                    expr=expr if expr else None,
                    output_fields=['*']
                )

                memories = []
                for hits in results:
                    for hit in hits:
                        memory_data = {
                            "memory_id": hit.entity.get("memory_id"),
                            "content": hit.entity.get("content"),
                            "memory_type": MemoryType(hit.entity.get("memory_type")),
                            "timestamp": hit.entity.get("timestamp"),
                            "importance": float(hit.entity.get("importance")),
                            "confidence": float(hit.entity.get("confidence")),
                            "tags": json.loads(hit.entity.get("tags")),
                            "emotions": json.loads(hit.entity.get("emotions")),
                            "source": hit.entity.get("source") if hit.entity.get("source") else None,
                            "context": json.loads(hit.entity.get("context")),
                            "embedding": hit.entity.get("embedding")
                        }
                        memories.append(MemoryItem(**memory_data))

            else:
                # Regular query without vector similarity
                results = self.collection.query(
                    expr=expr if expr else None,
                    output_fields=['*'],
                    limit=query.limit
                )

                memories = []
                for result in results:
                    memory_data = {
                        "memory_id": result["memory_id"],
                        "content": result["content"],
                        "memory_type": MemoryType(result["memory_type"]),
                        "timestamp": result["timestamp"],
                        "importance": float(result["importance"]),
                        "confidence": float(result["confidence"]),
                        "tags": json.loads(result["tags"]),
                        "emotions": json.loads(result["emotions"]),
                        "source": result["source"] if result["source"] else None,
                        "context": json.loads(result["context"]),
                        "embedding": result["embedding"]
                    }
                    memories.append(MemoryItem(**memory_data))

            return memories

        except Exception as e:
            logger.error(f"Failed to query memories: {e}")
            raise

    async def close(self):
        """Close the Milvus connection"""
        try:
            connections.disconnect("default")
            logger.info("Closed Milvus connection")
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {e}")