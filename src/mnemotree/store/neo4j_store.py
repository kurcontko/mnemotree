from __future__ import annotations

import json
import logging
import time
from asyncio import Lock
from typing import Any
from uuid import uuid4

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from pydantic import ValidationError

from ..core.models import MemoryItem
from ..core.query import MemoryQuery
from ..errors import StoreError
from ._records import build_neo4j_memory_payload, parse_neo4j_node_data
from ._schema import apply_neo4j_schema
from .base import BaseMemoryStore
from .logging import elapsed_ms, store_log_context
from .query_builders import build_neo4j_where_clause
from .serialization import serialize_datetime, serialize_datetime_list

logger = logging.getLogger(__name__)


class Neo4jMemoryStore(BaseMemoryStore):
    store_type = "neo4j"

    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._memory_locks: dict[str, Lock] = {}
        self._global_lock = Lock()
        self._init_lock = Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize database with required indexes and constraints"""
        async with self._init_lock:
            if self._initialized:
                return
            start = time.perf_counter()
            async with self.driver.session() as session:
                tx = await session.begin_transaction()
                try:
                    await apply_neo4j_schema(tx)

                    await tx.commit()
                    logger.info(
                        "Successfully initialized Neo4j schema",
                        extra=store_log_context(
                            self.store_type,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                    self._initialized = True
                except Neo4jError as e:
                    await tx.rollback()
                    logger.exception(
                        "Failed to initialize Neo4j schema",
                        extra=store_log_context(
                            self.store_type,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                    raise StoreError(
                        "Failed to initialize Neo4j schema",
                        store_type=self.store_type,
                        original_error=e,
                    ) from e

    async def _get_memory_lock(self, memory_id: str) -> Lock:
        """Get or create a lock for a specific memory"""
        async with self._global_lock:
            if memory_id not in self._memory_locks:
                self._memory_locks[memory_id] = Lock()
            return self._memory_locks[memory_id]

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a single memory with transaction safety"""
        await self.initialize()
        memory_lock = await self._get_memory_lock(memory.memory_id)

        async with memory_lock, self.driver.session() as session:
            tx = await session.begin_transaction()
            start = time.perf_counter()
            try:
                payload, valid_entities = build_neo4j_memory_payload(memory)

                # Store the memory node with all flattened properties (always).
                await tx.run(
                    """
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
                    """,
                    payload,
                )

                # Create relationships
                if memory.tags:
                    await tx.run(
                        """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $tags AS tag
                            MERGE (t:Tag {name: tag})
                            MERGE (m)-[:HAS_TAG]->(t)
                        """,
                        {"memory_id": memory.memory_id, "tags": memory.tags},
                    )

                # Handle associations
                if memory.associations:
                    await tx.run(
                        """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $associations AS assoc_id
                            MATCH (a:MemoryItem {memory_id: assoc_id})
                            MERGE (m)-[:ASSOCIATED_WITH]->(a)
                        """,
                        {"memory_id": memory.memory_id, "associations": memory.associations},
                    )

                # Handle conflicts
                if memory.conflicts_with:
                    await tx.run(
                        """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $conflicts AS conflict_id
                            MATCH (c:MemoryItem {memory_id: conflict_id})
                            MERGE (m)-[:CONFLICTS_WITH]->(c)
                        """,
                        {"memory_id": memory.memory_id, "conflicts": memory.conflicts_with},
                    )

                # Create entity references while keeping the data flattened
                if valid_entities:
                    await tx.run(
                        """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $entity_pairs as pair
                            MERGE (e:Entity {text: pair.text})
                            ON CREATE SET e.type = pair.type
                            MERGE (m)-[:MENTIONS_ENTITY]->(e)
                        """,
                        {
                            "memory_id": memory.memory_id,
                            "entity_pairs": [
                                {"text": text, "type": etype}
                                for text, etype in valid_entities.items()
                            ],
                        },
                    )

                # Handle temporal relationships
                if memory.previous_event_id:
                    await tx.run(
                        """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (p:MemoryItem {memory_id: $prev_id})
                            MERGE (p)-[r:TEMPORAL_NEXT {timestamp: $timestamp}]->(m)
                        """,
                        {
                            "memory_id": memory.memory_id,
                            "prev_id": memory.previous_event_id,
                            "timestamp": serialize_datetime(memory.timestamp),
                        },
                    )

                if memory.next_event_id:
                    await tx.run(
                        """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (n:MemoryItem {memory_id: $next_id})
                            MERGE (m)-[r:TEMPORAL_NEXT {timestamp: $timestamp}]->(n)
                        """,
                        {
                            "memory_id": memory.memory_id,
                            "next_id": memory.next_event_id,
                            "timestamp": serialize_datetime(memory.timestamp),
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

            except (Neo4jError, TypeError, ValueError) as e:
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
                raise StoreError(
                    "Failed to store memory in Neo4j",
                    store_type=self.store_type,
                    memory_id=memory.memory_id,
                    original_error=e,
                ) from e

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool:
        """
        Delete a memory item and optionally its relationships.

        Args:
            memory_id (str): The unique identifier of the memory.
            cascade (bool, optional): If True, delete related connections. Defaults to False.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        await self.initialize()
        async with self.driver.session() as session:
            start = time.perf_counter()
            try:
                if cascade:
                    # DETACH DELETE removes the node and all its relationships
                    result = await session.run(
                        """
                        MATCH (m:MemoryItem {memory_id: $memory_id})
                        DETACH DELETE m
                        RETURN COUNT(m) AS deleted
                    """,
                        memory_id=memory_id,
                    )
                else:
                    # Only delete the node if it has no relationships
                    result = await session.run(
                        """
                        MATCH (m:MemoryItem {memory_id: $memory_id})
                        WHERE NOT (m)--()
                        DELETE m
                        RETURN COUNT(m) AS deleted
                    """,
                        memory_id=memory_id,
                    )

                deleted = await result.single()
                success = deleted["deleted"] > 0
                if not success:
                    logger.info(
                        "No memory found to delete",
                        extra=store_log_context(
                            self.store_type,
                            memory_id=memory_id,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                    return False
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
            except Neo4jError as e:
                logger.exception(
                    "Failed to delete memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise StoreError(
                    "Failed to delete memory from Neo4j",
                    store_type=self.store_type,
                    memory_id=memory_id,
                    original_error=e,
                ) from e

    def _build_entity_query_memory(self, record: Any) -> MemoryItem:
        node_data = parse_neo4j_node_data(
            dict(record["m"]),
            strict_json=True,
        )
        context = node_data.get("context", {})
        if not isinstance(context, dict):
            raise TypeError("Expected context to be a dict.")
        context = dict(context)
        context.update(
            {
                "matching_entities": [
                    {"text": entity["text"], "type": entity["type"]}
                    for entity in record["matching_entities"]
                ],
                "connection_depth": record["shortest_path"],
            }
        )
        node_data["context"] = context
        return MemoryItem(**node_data)

    async def _collect_entity_query_results(
        self,
        result: Any,
        *,
        start: float,
        log_per_record: bool,
    ) -> list[MemoryItem]:
        memories: list[MemoryItem] = []
        parse_errors = 0
        last_error: Exception | None = None
        total_records = 0
        async for record in result:
            total_records += 1
            try:
                memories.append(self._build_entity_query_memory(record))
            except (ValidationError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                parse_errors += 1
                last_error = exc
                if log_per_record:
                    memory_id = dict(record.get("m", {})).get("memory_id", "unknown")
                    logger.warning(
                        "Failed to parse entity query record for memory %s: %s",
                        memory_id,
                        exc.__class__.__name__,
                        extra=store_log_context(
                            self.store_type,
                            memory_id=memory_id if memory_id != "unknown" else None,
                        ),
                    )
        if parse_errors:
            if log_per_record:
                logger.warning(
                    "Partial result: %d/%d records failed to parse in entity query",
                    parse_errors,
                    total_records,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                    exc_info=last_error,
                )
            else:
                logger.warning(
                    "Skipped %s entity records due to parse errors",
                    parse_errors,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                    exc_info=last_error,
                )
        return memories

    async def query_by_entities(
        self, entities: dict[str, str] | list[str], limit: int = 10, depth: int = 2
    ) -> list[MemoryItem]:
        """
        Query memories by mentioned entities with deep path traversal

        Args:
            entities: Dict mapping entity text to entity type, or a list of entity names (type-agnostic).
            limit: Maximum number of memories to return
            depth: Maximum path length to traverse
        """
        await self.initialize()
        start = time.perf_counter()
        try:
            if isinstance(entities, list):
                entity_names = [e for e in entities if e]
                async with self.driver.session() as session:
                    result = await session.run(
                        """
                        UNWIND $entity_names AS name
                        MATCH (e:Entity {text: name})
                        MATCH path = (e)<-[:MENTIONS_ENTITY*1..2]-(m:MemoryItem)
                        WITH m,
                             collect(DISTINCT {text: e.text, type: e.type}) as matching_entities,
                             min(length(path)) as shortest_path
                        ORDER BY size(matching_entities) DESC,
                                 shortest_path ASC,
                                 m.importance DESC
                        LIMIT $limit
                        RETURN m, matching_entities, shortest_path
                        """,
                        {"entity_names": entity_names, "limit": limit},
                    )
                    return await self._collect_entity_query_results(
                        result, start=start, log_per_record=True
                    )

            async with self.driver.session() as session:
                result = await session.run(
                    """
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
                """,
                    {
                        "entity_pairs": [
                            {"text": text, "type": etype}
                            for text, etype in (entities or {}).items()
                        ],
                        "limit": limit,
                    },
                )
                return await self._collect_entity_query_results(
                    result, start=start, log_per_record=False
                )
        except Neo4jError as e:
            logger.exception(
                "Failed to query by entities",
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise StoreError(
                "Failed to query memories by entities in Neo4j",
                store_type=self.store_type,
                original_error=e,
            ) from e

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory with transaction safety"""
        await self.initialize()
        async with self.driver.session() as session:
            # Begin transaction properly with await
            tx = await session.begin_transaction()
            start = time.perf_counter()
            try:
                result = await tx.run(
                    """
                    MATCH (m:MemoryItem {memory_id: $memory_id})
                    OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
                    OPTIONAL MATCH (m)-[:ASSOCIATED_WITH]->(a:MemoryItem)
                    RETURN m,
                           collect(DISTINCT t.name) as tags,
                           collect(DISTINCT a.memory_id) as associations
                """,
                    memory_id=memory_id,
                )

                record = await result.single()
                if not record:
                    await tx.commit()
                    return None

                node_data = parse_neo4j_node_data(
                    dict(record["m"]),
                    parse_entities=False,
                    parse_entity_mentions=False,
                )
                node_data["tags"] = record["tags"]
                node_data["associations"] = record["associations"]

                await tx.commit()
                return MemoryItem(**node_data)

            except (
                Neo4jError,
                ValidationError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                await tx.rollback()
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

    async def _create_query_embedding_node(
        self,
        session: Any,
        *,
        query_id: str,
        query_embedding: list[float],
        query: str,
    ) -> None:
        await session.run(
            """
                CREATE (:QueryEmbedding {
                    query_id: $query_id,
                    embedding: $embedding,
                    query: $query
                })
            """,
            {"query_id": query_id, "embedding": query_embedding, "query": query},
        )

    async def _cleanup_query_embedding_node(self, session: Any, query_id: str) -> None:
        await session.run(
            "MATCH (q:QueryEmbedding {query_id: $query_id}) DELETE q",
            query_id=query_id,
        )

    async def _run_similarity_query(
        self,
        session: Any,
        *,
        query_id: str,
        similarity_threshold: float,
        top_k: int,
    ) -> Any:
        return await session.run(
            """
                // First get similar memories
                    MATCH (q:QueryEmbedding {query_id: $query_id}), (m:MemoryItem)
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
            """,
            {"query_id": query_id, "threshold": similarity_threshold, "limit": top_k},
        )

    def _build_similarity_memory(self, memory_data: dict[str, Any], record: Any) -> MemoryItem:
        memory_data["tags"] = record["tags"]
        memory_data["associations"] = record["associations"]
        memory_data["conflicts_with"] = record["conflicts"]

        if record["next_id"]:
            memory_data["next_event_id"] = record["next_id"]
        if record["prev_id"]:
            memory_data["previous_event_id"] = record["prev_id"]

        memory_data = parse_neo4j_node_data(memory_data)

        if record["entity_data"]:
            memory_data["entities"].update(
                {ent["text"]: ent["type"] for ent in record["entity_data"]}
            )

        return MemoryItem(**memory_data)

    async def _collect_similarity_results(
        self,
        result: Any,
    ) -> tuple[list[MemoryItem], list[float], int, Exception | None, int]:
        memories: list[MemoryItem] = []
        scores: list[float] = []
        parse_errors = 0
        last_parse_error: Exception | None = None
        total_records = 0

        async for record in result:
            total_records += 1
            memory_data = dict(record["memory"])
            score = record["score"]
            scores.append(score)

            try:
                memory_item = self._build_similarity_memory(memory_data, record)
                memories.append(memory_item)
            except (ValidationError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                parse_errors += 1
                last_parse_error = exc
                memory_id = memory_data.get("memory_id", "unknown")
                logger.warning(
                    "Failed to parse similarity search record for memory %s: %s",
                    memory_id,
                    exc.__class__.__name__,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id if memory_id != "unknown" else None,
                    ),
                )

        return memories, scores, parse_errors, last_parse_error, total_records

    def _log_similarity_stats(
        self,
        *,
        memories: list[MemoryItem],
        scores: list[float],
        similarity_threshold: float,
        start: float,
    ) -> None:
        if scores:
            logger.info(
                "Found %s similar memories with scores: max=%.4f, min=%.4f, avg=%.4f",
                len(memories),
                max(scores),
                min(scores),
                sum(scores) / len(scores),
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )

            rel_stats = {
                "with_tags": sum(1 for memory in memories if memory.tags),
                "with_entities": sum(1 for memory in memories if memory.entities),
                "with_associations": sum(1 for memory in memories if memory.associations),
                "with_temporal": sum(
                    1 for memory in memories if memory.next_event_id or memory.previous_event_id
                ),
            }
            logger.info(
                "Relationship stats: %s",
                rel_stats,
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
        else:
            logger.warning(
                "No memories found above threshold %.4f",
                similarity_threshold,
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 15,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Get similar memories with additional relationship data."""
        await self.initialize()
        start = time.perf_counter()
        similarity_threshold: float = 0.1
        async with self.driver.session() as session:
            query_id = str(uuid4())
            await self._create_query_embedding_node(
                session, query_id=query_id, query_embedding=query_embedding, query=query
            )

            try:
                result = await self._run_similarity_query(
                    session,
                    query_id=query_id,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k,
                )

                (
                    memories,
                    scores,
                    parse_errors,
                    last_parse_error,
                    total_records,
                ) = await self._collect_similarity_results(result)

                if parse_errors:
                    logger.warning(
                        "Partial result: %d/%d records failed to parse in similarity search",
                        parse_errors,
                        total_records,
                        extra=store_log_context(
                            self.store_type,
                            duration_ms=elapsed_ms(start),
                        ),
                        exc_info=last_parse_error,
                    )

                self._log_similarity_stats(
                    memories=memories,
                    scores=scores,
                    similarity_threshold=similarity_threshold,
                    start=start,
                )
                return memories

            except Neo4jError as e:
                logger.exception(
                    "Failed to get similar memories",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise StoreError(
                    "Failed to retrieve similar memories from Neo4j",
                    store_type=self.store_type,
                    original_error=e,
                ) from e
            finally:
                await self._cleanup_query_embedding_node(session, query_id)

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
        await self.initialize()
        async with self.driver.session() as session:
            start = time.perf_counter()
            try:
                # 1. Update associations
                if related_ids is not None:
                    # Remove existing associations
                    await session.run(
                        """
                        MATCH (m:MemoryItem {memory_id: $memory_id})-[r:ASSOCIATED_WITH]->()
                        DELETE r
                    """,
                        memory_id=memory_id,
                    )

                    # Add new associations
                    if related_ids:
                        await session.run(
                            """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $related_ids AS related_id
                            MATCH (r:MemoryItem {memory_id: related_id})
                            MERGE (m)-[:ASSOCIATED_WITH]->(r)
                        """,
                            memory_id=memory_id,
                            related_ids=related_ids,
                        )

                # 2. Update conflicts
                if conflict_ids is not None:
                    # Remove existing conflicts
                    await session.run(
                        """
                        MATCH (m:MemoryItem {memory_id: $memory_id})-[r:CONFLICTS_WITH]->()
                        DELETE r
                    """,
                        memory_id=memory_id,
                    )

                    # Add new conflicts
                    if conflict_ids:
                        await session.run(
                            """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            UNWIND $conflict_ids AS conflict_id
                            MATCH (c:MemoryItem {memory_id: conflict_id})
                            MERGE (m)-[:CONFLICTS_WITH]->(c)
                        """,
                            memory_id=memory_id,
                            conflict_ids=conflict_ids,
                        )

                # 3. Update temporal relationships
                if previous_id is not None:
                    # Remove incoming temporal next to this memory (which means this is next of previous)
                    await session.run(
                        """
                        MATCH (p:MemoryItem)-[r:TEMPORAL_NEXT]->(m:MemoryItem {memory_id: $memory_id})
                        DELETE r
                    """,
                        memory_id=memory_id,
                    )

                    if previous_id:
                        await session.run(
                            """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (p:MemoryItem {memory_id: $previous_id})
                            MERGE (p)-[:TEMPORAL_NEXT]->(m)
                        """,
                            memory_id=memory_id,
                            previous_id=previous_id,
                        )

                if next_id is not None:
                    # Remove outgoing temporal next from this memory
                    await session.run(
                        """
                        MATCH (m:MemoryItem {memory_id: $memory_id})-[r:TEMPORAL_NEXT]->()
                        DELETE r
                    """,
                        memory_id=memory_id,
                    )

                    if next_id:
                        await session.run(
                            """
                            MATCH (m:MemoryItem {memory_id: $memory_id})
                            MATCH (n:MemoryItem {memory_id: $next_id})
                            MERGE (m)-[:TEMPORAL_NEXT]->(n)
                        """,
                            memory_id=memory_id,
                            next_id=next_id,
                        )

                logger.info(
                    "Successfully updated connections for memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
            except Neo4jError as e:
                logger.exception(
                    "Failed to update connections for memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise StoreError(
                    "Failed to update connections in Neo4j",
                    store_type=self.store_type,
                    memory_id=memory_id,
                    original_error=e,
                ) from e

    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        """Execute a complex memory query"""
        await self.initialize()
        # Note: This is a simplified implementation that supports basic filtering
        async with self.driver.session() as session:
            start = time.perf_counter()
            try:
                # Build MATCH and WHERE clauses
                cypher = "MATCH (m:MemoryItem)\n"
                params: dict[str, Any] = {}

                # Add vector similarity calculation if vector is present
                if query.vector:
                    # Neo4j vector index query usually requires a different syntax (CALL db.index.vector.queryNodes)
                    # For simplicity/compatibility we might assume brute force or pre-filtering.
                    # However, to properly integrate with filtering, we often filter first then sort, or vector search then filter.
                    # Here we will do valid filtering first as per cypher structure.
                    logger.debug(
                        "query_memories() does not currently support Neo4j vector search; ignoring query.vector",
                        extra=store_log_context(self.store_type),
                    )

                if query.relationships:
                    raise NotImplementedError(
                        "Neo4j relationship filtering is not supported by query_memories() yet."
                    )

                where_clause, where_params = build_neo4j_where_clause(query.filters or [])
                if where_clause:
                    cypher += where_clause + "\n"
                    params.update(where_params)

                # Limit
                cypher += f"RETURN m LIMIT {query.limit or 10}"

                result = await session.run(cypher, params)

                memories = []
                async for record in result:
                    node_data = parse_neo4j_node_data(dict(record["m"]))
                    # Note: we are missing relationships in this simple query
                    memories.append(MemoryItem(**node_data))

                return memories

            except (
                Neo4jError,
                ValidationError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                logger.exception(
                    "Failed to query memories",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def update_memory_metadata(self, memory_id: str, metadata: dict[str, Any]) -> bool:
        """Update metadata fields for a memory item."""
        await self.initialize()
        start = time.perf_counter()
        allowed_fields = {"last_accessed", "access_count", "access_history"}
        updates = {k: v for k, v in metadata.items() if k in allowed_fields}
        if not updates:
            return False

        set_clauses = []
        params: dict[str, Any] = {"memory_id": memory_id}
        for key, value in updates.items():
            if key == "last_accessed":
                value = serialize_datetime(value)
            elif key == "access_history":
                value = serialize_datetime_list(value)
            set_clauses.append(f"m.{key} = ${key}")
            params[key] = value

        cypher = f"""
            MATCH (m:MemoryItem {{memory_id: $memory_id}})
            SET {", ".join(set_clauses)}
            RETURN m
        """

        async with self.driver.session() as session:
            try:
                result = await session.run(cypher, params)
                record = await result.single()
                return record is not None
            except Neo4jError as e:
                logger.exception(
                    "Failed to update memory metadata for %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise StoreError(
                    "Failed to update memory metadata in Neo4j",
                    store_type=self.store_type,
                    memory_id=memory_id,
                    original_error=e,
                ) from e

    async def close(self):
        """Close the database connection"""
        await self.driver.close()
