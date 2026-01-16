from __future__ import annotations

import logging
import time
from asyncio import Lock
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from pydantic import ValidationError

from ...core.models import MemoryItem, MemoryType
from ...core.query import MemoryQuery
from ..base import BaseMemoryStore
from ..logging import elapsed_ms, store_log_context

logger = logging.getLogger(__name__)


class BaselineNeo4jMemoryStore(BaseMemoryStore):
    store_type = "neo4j_baseline"

    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._memory_locks = {}
        self._global_lock = Lock()

    async def initialize(self):
        """Initialize database with required indexes and constraints"""
        async with self.driver.session() as session:
            # Begin transaction properly with await
            tx = await session.begin_transaction()
            start = time.perf_counter()
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
                logger.info(
                    "Successfully initialized Neo4j schema",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
            except Neo4jError:
                await tx.rollback()
                logger.exception(
                    "Failed to initialize Neo4j schema",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
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

        async with memory_lock, self.driver.session() as session:  # Prevent concurrent modifications
            # Begin transaction properly with await
            tx = await session.begin_transaction()
            start = time.perf_counter()
            try:
                # Prepare data
                embedding_list = (
                    memory.embedding.tolist()
                    if isinstance(memory.embedding, np.ndarray)
                    else memory.embedding
                )

                # Store the memory node
                await tx.run(
                    """
                    CREATE (m:MemoryItem {
                        memory_id: $memory_id,
                        content: $content,
                        memory_type: $memory_type,
                        timestamp: $timestamp,
                        embedding: $embedding,
                        importance: $importance
                    })
                """,
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type.value,
                        "timestamp": memory.timestamp.isoformat()
                        if isinstance(memory.timestamp, datetime)
                        else str(memory.timestamp),
                        "embedding": embedding_list,
                        "importance": memory.importance,
                    },
                )

                await tx.commit()
                logger.info(
                    "Successfully stored memory %s",
                    memory.memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory.memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )

            except (Neo4jError, TypeError, ValueError):
                await tx.rollback()
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

    async def get_entity_contexts(self, _entities: list[str]) -> list[MemoryItem]:
        # Given a list of entity texts, find related memory items
        return []

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory with transaction safety"""
        return None

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool:
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
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Get similar memories using cosine similarity"""
        async with self.driver.session() as session:
            query_id = str(uuid4())
            # Create a temporary node with query embedding
            await session.run(
                """
                CREATE (:QueryEmbedding {query_id: $query_id, embedding: $embedding})
            """,
                {"query_id": query_id, "embedding": query_embedding},
            )

            # Calculate cosine similarity and get top results
            result = await session.run(
                """
                MATCH (q:QueryEmbedding {query_id: $query_id}), (m:MemoryItem)
                WITH m, gds.similarity.cosine(q.embedding, m.embedding) AS similarity
                ORDER BY similarity DESC
                LIMIT $limit
                RETURN m {
                    .memory_id, .content, .memory_type, .embedding, .importance
                } AS memory,
                similarity
            """,
                {"query_id": query_id, "limit": top_k},
            )

            # Clean up query node
            await session.run(
                "MATCH (q:QueryEmbedding {query_id: $query_id}) DELETE q",
                query_id=query_id,
            )

            # Convert results to MemoryItems
            memories = []
            async for record in result:
                memory_data = record["memory"]
                try:
                    memory_type = MemoryType(memory_data["memory_type"])
                    memory_item = MemoryItem(
                        memory_id=memory_data["memory_id"],
                        content=memory_data["content"],
                        memory_type=memory_type,
                        embedding=memory_data["embedding"],
                        importance=memory_data["importance"],
                    )
                except ValidationError as e:
                    logger.error(
                        "Validation error when creating MemoryItem",
                        extra=store_log_context(
                            self.store_type,
                            memory_id=memory_data.get("memory_id"),
                        ),
                    )
                    for error in e.errors():
                        logger.error(
                            "Field: %s, Error: %s, Input value: %s, Expected type: %s",
                            " -> ".join(error["loc"]),
                            error["msg"],
                            error.get("input", "N/A"),
                            error.get("type", "N/A"),
                            extra=store_log_context(
                                self.store_type,
                                memory_id=memory_data.get("memory_id"),
                            ),
                        )
                    raise
                memories.append(memory_item)

            return memories

    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: list[str] | None = None,
        conflict_ids: list[str] | None = None,
        previous_id: str | None = None,
        next_id: str | None = None,
    ) -> None:
        """Update memory connections"""
        raise NotImplementedError("Method not implemented")

    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        """Execute a complex memory query"""
        raise NotImplementedError("Method not implemented")

    async def close(self):
        """Close the database connection"""
        await self.driver.close()
