from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

from chromadb.api.models.Collection import Collection

try:
    from chromadb.errors import ChromaError
except ImportError:  # pragma: no cover - compatibility fallback
    ChromaError = Exception

from ..core.models import MemoryItem
from ..core.query import MemoryQuery
from ..errors import IndexError as MnemotreeIndexError
from ..errors import StoreError
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
        if self.collection is None:
            return []
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
            if self.collection is not None:
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
                "Failed to store memory in ChromaDB",
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
            if self.collection is None:
                return None
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
            if self.collection is not None:
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
            if self.collection is None:
                return []
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
            if self.collection is None:
                return []
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

    def _query_graph_index_hits(
        self,
        entities: dict[str, str] | list[str],
        *,
        limit: int,
        start: float,
    ) -> list[Any] | None:
        if not self.graph_index:
            return None
        try:
            hits = self.graph_index.recall_by_entities(
                entities,
                limit=limit,
                hops=self.graph_hops,
            )
        except Exception as exc:
            logger.exception(
                "Failed to query graph index: %s",
                exc.__class__.__name__,
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            if self.graph_index_strict:
                raise MnemotreeIndexError(
                    "Failed to query graph index by entities"
                ) from exc
            return None
        if not hits:
            return None
        return hits

    async def _hydrate_graph_hits(
        self,
        hits: list[Any],
        *,
        limit: int,
    ) -> list[MemoryItem]:
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

    def _scan_collection_for_entities(
        self,
        entities: dict[str, str] | list[str],
        *,
        limit: int,
    ) -> list[MemoryItem]:
        entity_set = build_entity_set(entities)
        if self.collection is None:
            return []
        all_results = self.collection.get(include=["embeddings", "documents", "metadatas"])

        if not all_results["ids"]:
            return []

        matching_memories: list[MemoryItem] = []
        for idx, memory_id in enumerate(all_results["ids"]):
            metadata = all_results["metadatas"][idx]
            stored_entities = json_loads_dict(metadata.get("entities"))

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

    def _reset_graph_index(self) -> bool:
        if self.graph_index is None:
            return True
        try:
            self.graph_index.reset()
        except Exception as exc:
            logger.exception(
                "Failed to reset graph index: %s",
                exc.__class__.__name__,
                extra=store_log_context(self.store_type),
            )
            if self.graph_index_strict:
                raise MnemotreeIndexError(
                    "Failed to reset graph index during rebuild"
                ) from exc
            return False
        return True

    def _iter_collection_batches(self, *, batch_size: int) -> Iterator[dict[str, Any]]:
        if self.collection is None:
            return
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
                    return
                batch = self.collection.get(include=["embeddings", "documents", "metadatas"])
            ids = batch.get("ids") or []
            if not ids:
                return
            yield batch
            if len(ids) < batch_size:
                return
            offset += len(ids)

    def _upsert_graph_batch(self, batch: dict[str, Any]) -> int:
        if self.graph_index is None:
            return 0
        ids = batch.get("ids") or []
        rebuilt = 0
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
            except Exception as exc:
                logger.warning(
                    "Failed to index memory %s during rebuild: %s",
                    memory_id,
                    exc.__class__.__name__,
                    extra=store_log_context(self.store_type, memory_id=memory_id),
                )
                if self.graph_index_strict:
                    raise MnemotreeIndexError(
                        f"Failed to rebuild graph index for memory {memory_id}"
                    ) from exc
        return rebuilt

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

            hits = self._query_graph_index_hits(entities, limit=limit, start=start)
            if hits is not None:
                return await self._hydrate_graph_hits(hits, limit=limit)

            return self._scan_collection_for_entities(entities, limit=limit)

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

        if not self._reset_graph_index():
            return 0

        rebuilt = 0
        batch_size = 1000
        for batch in self._iter_collection_batches(batch_size=batch_size):
            rebuilt += self._upsert_graph_batch(batch)
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
        if self.collection is None:
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
