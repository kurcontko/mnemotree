from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from chromadb.api.models.Collection import Collection

try:
    from chromadb.errors import ChromaError
except ImportError:  # pragma: no cover - compatibility fallback
    ChromaError = Exception

from ..core.models import MemoryItem
from ..core.query import MemoryQuery
from ..errors import IndexError as MnemotreeIndexError, StoreError
from ..utils.serialization import json_loads_dict
from ._queries import build_entity_set, entity_matches
from ._records import chroma_memory_from_record, chroma_metadata_from_memory
from .base import BaseMemoryStore
from .chroma_utils import create_chroma_client
from .logging import elapsed_ms, store_log_context
from .query_builders import UnsupportedQueryError, build_chroma_where
from .sqlite_graph import SQLiteGraphIndex

logger = logging.getLogger(__name__)




class ChromaMemoryStore(BaseMemoryStore):
    store_type = "chroma"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        ssl: bool = False,
        persist_directory: str | None = None,
        collection_name: str = "memories",
        headers: dict[str, str] | None = None,
        *,
        enable_graph_index: bool = True,
        graph_db_path: str | None = None,
        graph_hops: int = 2,
        graph_index_strict: bool = False,
    ):
        """Initialize ChromaDB storage with either local persistence or remote connection.
        Selected based on provided arguments. Provide either host/port for remote connection or persist_directory for local persistence.

        Args:
            host: Optional host address for remote ChromaDB instance (default: None) e.g. "localhost"
            port: Optional port for remote ChromaDB instance (default: None) e.g. 8000
            persist_directory: Optional directory for local persistence e.g. ".mnemotree/chromadb-local"
            collection_name: Name of the collection to use
            ssl: Whether to use SSL for remote connection
            headers: Optional headers for authentication/authorization
            enable_graph_index: Whether to maintain a local SQLite graph index
            graph_db_path: Optional path for the graph index SQLite DB
            graph_hops: Number of hops to traverse in graph-based entity queries
            graph_index_strict: If True, fail memory operations when graph indexing fails
        """
        self.client = create_chroma_client(
            host=host,
            port=port,
            ssl=ssl,
            persist_directory=persist_directory,
            headers=headers,
            store_type=self.store_type,
        )

        self.collection_name = collection_name
        self.collection: Collection | None = None
        self._initialized = False
        self.graph_hops = max(1, int(graph_hops))
        self.graph_index_strict = bool(graph_index_strict)
        self.graph_index: SQLiteGraphIndex | None = None
        if enable_graph_index:
            resolved_graph_path = graph_db_path
            if resolved_graph_path is None and persist_directory:
                resolved_graph_path = os.path.join(persist_directory, "lite_graph.sqlite3")
            if resolved_graph_path:
                self.graph_index = SQLiteGraphIndex(resolved_graph_path)

    async def initialize(self):
        """Initialize or get the memories collection"""
        if self._initialized:
            return
        start = time.perf_counter()
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
            logger.info(
                "Successfully initialized ChromaDB collection: %s",
                self.collection_name,
                extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start)),
            )
            if self.graph_index:
                try:
                    self.graph_index.initialize()
                except Exception as e:
                    logger.exception(
                        "Failed to initialize graph index",
                        extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start)),
                    )
                    if self.graph_index_strict:
                        raise MnemotreeIndexError(
                            f"Failed to initialize graph index: {e}"
                        ) from e
                    logger.warning(
                        "Continuing without graph index due to initialization failure",
                        extra=store_log_context(self.store_type),
                    )
                    self.graph_index = None
            self._initialized = True
        except ChromaError as e:
            logger.exception(
                "Failed to initialize ChromaDB collection: %s",
                self.collection_name,
                extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start)),
            )
            raise StoreError(
                f"Failed to initialize ChromaDB collection: {self.collection_name}",
                store_type=self.store_type,
                original_error=e,
            ) from e

    async def list_memories(
        self,
        *,
        include_embeddings: bool = False,
    ) -> list[MemoryItem]:
        """Return all memories stored in the collection."""
        await self.initialize()
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        results = self.collection.get(include=include)
        ids = results.get("ids") or []
        if not ids:
            return []
        embeddings = results.get("embeddings")
        memories: list[MemoryItem] = []
        for idx, memory_id in enumerate(ids):
            memories.append(
                chroma_memory_from_record(
                    memory_id=memory_id,
                    document=results["documents"][idx],
                    embedding=embeddings[idx] if embeddings is not None else None,
                    metadata=results["metadatas"][idx],
                )
            )
        return memories

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a single memory"""
        await self.initialize()
        start = time.perf_counter()
        try:
            metadata = chroma_metadata_from_memory(memory)

            # Store in ChromaDB
            self.collection.upsert(
                ids=[memory.memory_id],
                embeddings=[memory.embedding],
                documents=[memory.content],
                metadatas=[metadata],
            )
            
            await self._update_graph_index_safe(memory, start)

            logger.info(
                "Successfully stored memory %s",
                memory.memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory.memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )

        except ChromaError as e:
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
                f"Failed to store memory in ChromaDB",
                store_type=self.store_type,
                memory_id=memory.memory_id,
                original_error=e,
            ) from e

    async def _update_graph_index_safe(self, memory: MemoryItem, start_time: float) -> None:
        """Safely update the graph index, handling errors based on strictness setting."""
        if not self.graph_index:
            return

        try:
            self.graph_index.upsert_memory(memory)
        except Exception as e:
            logger.exception(
                "Failed to update graph index for memory %s: %s",
                memory.memory_id,
                e.__class__.__name__,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory.memory_id,
                    duration_ms=elapsed_ms(start_time),
                ),
            )
            if self.graph_index_strict:
                raise MnemotreeIndexError(
                    f"Failed to update graph index for memory {memory.memory_id}"
                ) from e

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory by ID"""
        await self.initialize()
        start = time.perf_counter()
        try:
            result = self.collection.get(
                ids=[memory_id], include=["embeddings", "documents", "metadatas"]
            )

            if not result["ids"]:
                return None

            return chroma_memory_from_record(
                memory_id=memory_id,
                document=result["documents"][0],
                embedding=result["embeddings"][0],
                metadata=result["metadatas"][0],
            )

        except (ChromaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
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

        Note: Chroma is a pure vector store here; `cascade` is accepted for
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
            self.collection.delete(ids=[memory_id])
            if self.graph_index:
                try:
                    self.graph_index.delete_memory(memory_id)
                except Exception as e:
                    logger.exception(
                        "Failed to update graph index for deleted memory %s: %s",
                        memory_id,
                        e.__class__.__name__,
                        extra=store_log_context(
                            self.store_type,
                            memory_id=memory_id,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                    if self.graph_index_strict:
                        raise MnemotreeIndexError(
                            f"Failed to delete memory from graph index: {memory_id}"
                        ) from e
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
        except (ChromaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
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
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Get similar memories using vector similarity"""
        await self.initialize()
        start = time.perf_counter()
        try:
            results = self.collection.query(
                query_texts=[query],
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["embeddings", "documents", "metadatas"],
            )

            memories = []
            for i, memory_id in enumerate(results["ids"][0]):
                memories.append(
                    chroma_memory_from_record(
                        memory_id=memory_id,
                        document=results["documents"][0][i],
                        embedding=results["embeddings"][0][i],
                        metadata=results["metadatas"][0][i],
                    )
                )

            return memories

        except (ChromaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
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
            # Build where clause for metadata filtering
            where: dict[str, str] = {}
            if query.filters:
                where = build_chroma_where(query.filters)

            # Get results using vector similarity if provided, otherwise use metadata filtering
            results = self.collection.query(
                query_embeddings=[query.vector] if query.vector is not None else None,
                where=where if where else None,
                n_results=query.limit if query.limit else 10,
                include=["embeddings", "documents", "metadatas"],
            )

            # Convert results to MemoryItems
            memories = []
            for i, memory_id in enumerate(results["ids"][0]):
                memories.append(
                    chroma_memory_from_record(
                        memory_id=memory_id,
                        document=results["documents"][0][i],
                        embedding=results["embeddings"][0][i],
                        metadata=results["metadatas"][0][i],
                    )
                )

            return memories

        except UnsupportedQueryError:
            raise
        except (ChromaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to query memories",
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

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
        start = time.perf_counter()
        try:
            # 1. Get existing memory to preserve other fields
            memory = await self.get_memory(memory_id)
            if not memory:
                logger.warning(
                    "Attempted to update connections for non-existent memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return

            # 2. Update fields if provided
            if related_ids is not None:
                memory.associations = related_ids
            if conflict_ids is not None:
                memory.conflicts_with = conflict_ids
            if previous_id is not None:
                memory.previous_event_id = previous_id
            if next_id is not None:
                memory.next_event_id = next_id

            # 3. Re-store the memory (this will update metadata)
            await self.store_memory(memory)
            logger.info(
                "Successfully updated connections for memory %s",
                memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )

        except (ChromaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to update connections for memory %s",
                memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Query memories by entity names.

        Args:
            entities: List of entity names (or a dict mapping entity->type; types ignored in Chroma)
            limit: Maximum number of results to return

        Returns:
            List of MemoryItem objects containing any of the specified entities
        """
        await self.initialize()
        start = time.perf_counter()
        try:
            if not entities:
                return []

            if self.graph_index:
                try:
                    hits = self.graph_index.recall_by_entities(
                        entities,
                        limit=limit,
                        hops=self.graph_hops,
                    )
                except Exception as e:
                    logger.exception(
                        "Failed to query graph index: %s",
                        e.__class__.__name__,
                        extra=store_log_context(
                            self.store_type,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                    if self.graph_index_strict:
                        raise MnemotreeIndexError(
                            f"Failed to query graph index by entities"
                        ) from e
                    hits = []
                if not hits:
                    # Fall back to slower scan when the index is empty/unavailable.
                    hits = None
            else:
                hits = None

            if hits is not None:
                if not hits:
                    return []
                memories_by_id = await self._get_memories_by_ids(
                    [hit.memory_id for hit in hits]
                )
                results: list[MemoryItem] = []
                for hit in hits:
                    memory = memories_by_id.get(hit.memory_id)
                    if memory is None:
                        continue
                    context = memory.context if isinstance(memory.context, dict) else {}
                    if hit.matching_entities:
                        context = dict(context)
                        context["matching_entities"] = hit.matching_entities
                    if hit.depth:
                        context = dict(context)
                        context["connection_depth"] = hit.depth
                    if context:
                        memory.context = context
                    results.append(memory)
                    if len(results) >= limit:
                        break
                return results

            entity_set = build_entity_set(entities)

            # Get all memories and filter by entities in memory
            # ChromaDB doesn't support complex metadata queries on JSON fields,
            # so we need to retrieve and filter in memory
            all_results = self.collection.get(include=["embeddings", "documents", "metadatas"])

            if not all_results["ids"]:
                return []

            matching_memories = []

            for idx, memory_id in enumerate(all_results["ids"]):
                metadata = all_results["metadatas"][idx]
                stored_entities = json_loads_dict(metadata.get("entities"))

                # Check if any of the requested entities are in this memory
                if entity_matches(stored_entities, entity_set):
                    matching_memories.append(
                        chroma_memory_from_record(
                            memory_id=memory_id,
                            document=all_results["documents"][idx],
                            embedding=all_results["embeddings"][idx],
                            metadata=metadata,
                            entities_override=stored_entities,
                        )
                    )

                    if len(matching_memories) >= limit:
                        break

            return matching_memories

        except (ChromaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.exception(
                "Failed to query by entities",
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def rebuild_graph_index(self) -> int:
        """Rebuild the SQLite graph index from the current Chroma collection."""
        if not self.graph_index:
            return 0
        await self.initialize()
        assert self.collection is not None

        try:
            self.graph_index.reset()
        except Exception as e:
            logger.exception(
                "Failed to reset graph index: %s",
                e.__class__.__name__,
                extra=store_log_context(self.store_type),
            )
            if self.graph_index_strict:
                raise MnemotreeIndexError(
                    "Failed to reset graph index during rebuild"
                ) from e
            return 0

        rebuilt = 0
        batch_size = 1000
        offset = 0
        while True:
            try:
                batch = self.collection.get(
                    include=["embeddings", "documents", "metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
            except TypeError:
                if offset != 0:
                    break
                batch = self.collection.get(include=["embeddings", "documents", "metadatas"])
            ids = batch.get("ids") or []
            if not ids:
                break
            for idx, memory_id in enumerate(ids):
                memory = chroma_memory_from_record(
                    memory_id=memory_id,
                    document=batch["documents"][idx],
                    embedding=batch["embeddings"][idx],
                    metadata=batch["metadatas"][idx],
                )
                try:
                    self.graph_index.upsert_memory(memory)
                    rebuilt += 1
                except Exception as e:
                    logger.warning(
                        "Failed to index memory %s during rebuild: %s",
                        memory_id,
                        e.__class__.__name__,
                        extra=store_log_context(self.store_type, memory_id=memory_id),
                    )
                    if self.graph_index_strict:
                        raise MnemotreeIndexError(
                            f"Failed to rebuild graph index for memory {memory_id}"
                        ) from e
            if len(ids) < batch_size:
                break
            offset += len(ids)
        return rebuilt

    async def close(self):
        """Close the database connection"""
        # ChromaDB handles connection cleanup automatically
        self.collection = None
        self._initialized = False
        if self.graph_index:
            self.graph_index.close()

    async def _get_memories_by_ids(self, memory_ids: list[str]) -> dict[str, MemoryItem]:
        if not memory_ids:
            return {}
        result = self.collection.get(
            ids=memory_ids, include=["embeddings", "documents", "metadatas"]
        )
        memories: dict[str, MemoryItem] = {}
        for idx, memory_id in enumerate(result["ids"]):
            memories[memory_id] = chroma_memory_from_record(
                memory_id=memory_id,
                document=result["documents"][idx],
                embedding=result["embeddings"][idx],
                metadata=result["metadatas"][idx],
            )
        return memories
