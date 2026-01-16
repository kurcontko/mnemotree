from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

try:
    from pymilvus.exceptions import MilvusException
except ImportError:  # pragma: no cover - compatibility fallback
    MilvusException = Exception

from ..core.models import MemoryItem, MemoryType
from ..core.query import MemoryQuery
from ..utils.serialization import json_dumps_safe, json_loads_dict
from .base import BaseMemoryStore
from .logging import elapsed_ms, store_log_context

logger = logging.getLogger(__name__)


class MilvusMemoryStore(BaseMemoryStore):
    store_type = "milvus"

    def __init__(
        self,
        uri: str = "localhost:19530",
        user: str = "",
        password: str = "",
        collection_name: str = "memories",
        dim: int = 1536,  # Default dimension for embeddings
        consistency_level: str = "Strong",
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
        self.collection: Collection | None = None
        self._initialized = False

    async def initialize(self):
        """Initialize Milvus connection and create collection if needed"""
        if self._initialized:
            return
        start = time.perf_counter()
        try:
            # Connect to Milvus
            connections.connect(
                alias="default", uri=self.uri, user=self.user, password=self.password
            )
            logger.info(
                "Connected to Milvus at %s",
                self.uri,
                extra=store_log_context(self.store_type),
            )

            # Define collection schema
            fields = [
                FieldSchema(
                    name="memory_id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True
                ),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="importance", dtype=DataType.FLOAT),
                FieldSchema(name="confidence", dtype=DataType.FLOAT),
                FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="emotions", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]

            schema = CollectionSchema(fields=fields, description="Memory storage collection")

            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level,
                )

                # Create index for vector similarity search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
            else:
                self.collection = Collection(self.collection_name)
                self.collection.load()

            logger.info(
                "Successfully initialized Milvus collection: %s",
                self.collection_name,
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            self._initialized = True

        except MilvusException:
            logger.exception(
                "Failed to initialize Milvus",
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a single memory"""
        await self.initialize()
        start = time.perf_counter()
        try:
            timestamp = memory.timestamp
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()

            # Prepare data for insertion
            data = {
                "memory_id": [memory.memory_id],
                "content": [memory.content],
                "memory_type": [memory.memory_type.value],
                "timestamp": [timestamp],
                "importance": [float(memory.importance)],
                "confidence": [float(memory.confidence)],
                "tags": [json.dumps(memory.tags)],
                "emotions": [json.dumps([str(e) for e in memory.emotions])],
                "source": [memory.source if memory.source else ""],
                "context": [json_dumps_safe(memory.context) if memory.context else "{}"],
                "embedding": [memory.embedding],
            }

            # Insert into Milvus
            self.collection.insert(data)
            self.collection.flush()
            logger.info(
                "Successfully stored memory %s",
                memory.memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory.memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )

        except (MilvusException, TypeError, ValueError):
            logger.exception(
                "Failed to store memory %s",
                memory.memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory.memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory by ID"""
        await self.initialize()
        start = time.perf_counter()
        try:
            # Query by memory_id
            self.collection.load()
            results = self.collection.query(expr=f'memory_id == "{memory_id}"', output_fields=["*"])

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
                "context": json_loads_dict(result.get("context")),
                "embedding": result["embedding"],
            }

            return MemoryItem(**memory_data)

        except (MilvusException, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to retrieve memory %s",
                memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def delete_memory(
        self,
        memory_id: str,
        *,
        cascade: bool = False,
    ) -> bool:
        """Delete a memory by ID.

        Note: Milvus is a pure vector store here; `cascade` is accepted for
        interface compatibility but has no additional effect.
        """
        await self.initialize()
        start = time.perf_counter()
        try:
            memory = await self.get_memory(memory_id)
            if not memory:
                logger.info(
                    "No memory found to delete",
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return False
            expr = f'memory_id == "{memory_id}"'
            self.collection.delete(expr)
            logger.info(
                "Deleted memory %s",
                memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )
            return True
        except (MilvusException, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to delete memory %s",
                memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Get similar memories using vector similarity"""
        await self.initialize()
        start = time.perf_counter()
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
                output_fields=["*"],
            )

            return self._process_search_results(results)

        except (MilvusException, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to get similar memories",
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        """Execute a complex memory query"""
        await self.initialize()
        start = time.perf_counter()
        try:
            expr = self._build_milvus_expression(query)

            # If vector similarity is requested
            if query.vector is not None:
                return self._search_vector_query(query, expr)

            return self._search_scalar_query(query, expr)

        except (MilvusException, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to query memories",
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    def _build_milvus_expression(self, query: MemoryQuery) -> str:
        expressions = []
        if query.filters:
            for filter in query.filters:
                if isinstance(filter.value, str):
                    expressions.append(f'{filter.field} == "{filter.value}"')
                else:
                    expressions.append(f"{filter.field} {filter.operator.value} {filter.value}")
        return " && ".join(expressions) if expressions else ""

    def _search_vector_query(self, query: MemoryQuery, expr: str) -> list[MemoryItem]:
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
            output_fields=["*"],
        )
        return self._process_search_results(results)

    def _search_scalar_query(self, query: MemoryQuery, expr: str) -> list[MemoryItem]:
        results = self.collection.query(
            expr=expr if expr else None, output_fields=["*"], limit=query.limit
        )
        return [self._parse_memory_record(result) for result in results]

    def _process_search_results(self, results: Any) -> list[MemoryItem]:
        memories = []
        for hits in results:
            for hit in hits:
                memories.append(self._parse_memory_record(hit.entity))
        return memories

    def _parse_memory_record(self, record: Any) -> MemoryItem:
        # Handle difference between search result (entity object/dict-like) and query result (dict)
        # Assuming record acts like a dict in both cases or normalized before calling
        get_field = record.get if isinstance(record, dict) else lambda k: record.get(k)

        memory_data = {
            "memory_id": get_field("memory_id"),
            "content": get_field("content"),
            "memory_type": MemoryType(get_field("memory_type")),
            "timestamp": get_field("timestamp"),
            "importance": float(get_field("importance")),
            "confidence": float(get_field("confidence")),
            "tags": json.loads(get_field("tags")),
            "emotions": json.loads(get_field("emotions")),
            "source": get_field("source") if get_field("source") else None,
            "context": json_loads_dict(get_field("context")),
            "embedding": get_field("embedding"),
        }
        return MemoryItem(**memory_data)

    async def close(self):
        """Close the Milvus connection"""
        try:
            connections.disconnect("default")
            logger.info(
                "Closed Milvus connection",
                extra=store_log_context(self.store_type),
            )
            self.collection = None
            self._initialized = False
        except MilvusException:
            logger.exception(
                "Error closing Milvus connection",
                extra=store_log_context(self.store_type),
            )
