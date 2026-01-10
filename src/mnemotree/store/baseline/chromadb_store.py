from __future__ import annotations

import logging
import time
from typing import Any

from chromadb.api.models.Collection import Collection

try:
    from chromadb.errors import ChromaError
except ImportError:  # pragma: no cover - compatibility fallback
    ChromaError = Exception

from ...core.models import MemoryItem, MemoryType
from ...core.query import MemoryQuery
from ..base import BaseMemoryStore
from ..chroma_utils import create_chroma_client
from ..logging import elapsed_ms, store_log_context

logger = logging.getLogger(__name__)


class BaselineChromaStore(BaseMemoryStore):
    """ChromaDB baseline using only embeddings"""

    store_type = "chroma_baseline"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        ssl: bool = False,
        persist_directory: str | None = None,
        collection_name: str = "memories",
        headers: dict[str, str] | None = None,
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

    async def initialize(self):
        """Initialize collection"""
        start = time.perf_counter()
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                "Successfully initialized ChromaDB collection: %s",
                self.collection_name,
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
        except ChromaError:
            logger.exception(
                "Failed to initialize ChromaDB collection: %s",
                self.collection_name,
                extra=store_log_context(
                    self.store_type,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store memory with only embeddings"""
        self.collection.add(
            ids=[memory.memory_id],
            embeddings=[memory.embedding],
            documents=[memory.content],
            metadatas=[{"memory_id": memory.memory_id}],
        )

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Retrieve memory by ID"""
        result = self.collection.get(ids=[memory_id], include=["embeddings", "documents"])

        if not result["ids"]:
            return None

        return MemoryItem(
            memory_id=memory_id,
            content=result["documents"][0],
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            embedding=result["embeddings"][0],
        )

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool:
        """Delete a memory by ID.

        Note: This baseline store does not manage graph connections; `cascade` is
        accepted for interface compatibility but has no additional effect.
        """
        if self.collection is None:
            await self.initialize()
        try:
            existing = await self.get_memory(memory_id)
            if not existing:
                return False
            self.collection.delete(ids=[memory_id])
            return True
        except ChromaError:
            logger.exception(
                "Failed to delete memory %s",
                memory_id,
                extra=store_log_context(self.store_type, memory_id=memory_id),
            )
            raise

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Get similar memories using only vector similarity"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["embeddings", "documents", "metadatas"],
        )

        memories = []
        for i, memory_id in enumerate(results["ids"][0]):
            memories.append(
                MemoryItem(
                    memory_id=memory_id,
                    content=results["documents"][0][i],
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.5,
                    embedding=results["embeddings"][0][i],
                )
            )

        return memories

    async def update_connections(
        self,
        memory_id: str,
        related_ids: list[str] | None = None,
        conflict_ids: list[str] | None = None,
        previous_id: str | None = None,
        next_id: str | None = None,
    ) -> None:
        """Update memory connections"""
        raise NotImplementedError("Connections are not supported in ChromaDB")

    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        """Execute a complex memory query"""
        raise NotImplementedError("Query operations are not supported in ChromaDB")

    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Query memories by entities.

        This baseline store does not build an entity index. Returning an empty
        list keeps higher-level retrieval strategies (e.g. NER+RRF) runnable
        while preserving the "vector-only" baseline semantics.
        """
        return []

    async def close(self):
        """Close the database connection gracefully."""
        # ChromaDB client manages its own resources; clearing references is enough.
        self.collection = None
