import logging
from typing import List, Optional, Dict, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from ..core.models import MemoryItem, MemoryType
from ..core.query import MemoryQuery
from .base import BaseMemoryStore

logger = logging.getLogger(__name__)


class ChromaMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ssl: bool = False,
        persist_directory: Optional[str] = None,
        collection_name: str = "memories",
        headers: Optional[Dict[str, str]] = None,
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
        if host and port:
            # Connect to remote ChromaDB instance
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers or {}
            )
            logger.info(f"Initialized remote ChromaDB client at {host}:{port}")
        elif persist_directory:
            # Use local persistence
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized local ChromaDB client at {persist_directory}")
        else:
            # In-memory client for testing
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            logger.info("Initialized in-memory ChromaDB client")
            
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None

    async def initialize(self):
        """Initialize or get the memories collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Successfully initialized ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a single memory"""
        try:
            # Prepare metadata
            metadata = {
                "memory_type": memory.memory_type.value,
                "timestamp": memory.timestamp,
                "importance": str(memory.importance),  # ChromaDB requires string values
                "tags": ",".join(memory.tags) if memory.tags else "",
                "emotions": ",".join(str(e) for e in memory.emotions) if memory.emotions else "",
                "confidence": str(memory.confidence),
                "source": memory.source if memory.source else "",
                "context": str(memory.context) if memory.context else ""
            }

            # Store in ChromaDB
            self.collection.upsert(
                ids=[memory.memory_id],
                embeddings=[memory.embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            logger.info(f"Successfully stored memory {memory.memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory.memory_id}: {e}")
            raise

    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID"""
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["embeddings", "documents", "metadatas"]
            )

            if not result["ids"]:
                return None

            # Extract data from the first (and only) result
            metadata = result["metadatas"][0]
            
            # Convert metadata back to appropriate types
            memory_data = {
                "memory_id": memory_id,
                "content": result["documents"][0],
                "memory_type": MemoryType(metadata["memory_type"]),
                "timestamp": metadata["timestamp"],
                "importance": float(metadata["importance"]),
                "confidence": float(metadata["confidence"]),
                "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                "emotions": metadata["emotions"].split(",") if metadata["emotions"] else [],
                "source": metadata["source"] if metadata["source"] else None,
                "context": eval(metadata["context"]) if metadata["context"] else {},
                "embedding": result["embeddings"][0]
            }

            return MemoryItem(**memory_data)

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Get similar memories using vector similarity"""
        try:
            results = self.collection.query(
                query_texts=[query],
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["embeddings", "documents", "metadatas"]
            )

            memories = []
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                memory_data = {
                    "memory_id": memory_id,
                    "content": results["documents"][0][i],
                    "memory_type": MemoryType(metadata["memory_type"]),
                    "timestamp": metadata["timestamp"],
                    "importance": float(metadata["importance"]),
                    "confidence": float(metadata["confidence"]),
                    "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                    "emotions": metadata["emotions"].split(",") if metadata["emotions"] else [],
                    "source": metadata["source"] if metadata["source"] else None,
                    "context": eval(metadata["context"]) if metadata["context"] else {},
                    "embedding": results["embeddings"][0][i]
                }
                
                memories.append(MemoryItem(**memory_data))

            return memories

        except Exception as e:
            logger.error(f"Failed to get similar memories: {e}")
            raise

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Execute a complex memory query"""
        try:
            # Build where clause for metadata filtering
            where = {}
            if query.filters:
                for filter in query.filters:
                    where[filter.field] = str(filter.value)  # ChromaDB requires string values

            # Get results using vector similarity if provided, otherwise use metadata filtering
            results = self.collection.query(
                query_embeddings=[query.vector] if query.vector is not None else None,
                where=where if where else None,
                n_results=query.limit if query.limit else 10,
                include=["embeddings", "documents", "metadatas"]
            )

            # Convert results to MemoryItems
            memories = []
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                memory_data = {
                    "memory_id": memory_id,
                    "content": results["documents"][0][i],
                    "memory_type": MemoryType(metadata["memory_type"]),
                    "timestamp": metadata["timestamp"],
                    "importance": float(metadata["importance"]),
                    "confidence": float(metadata["confidence"]),
                    "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                    "emotions": metadata["emotions"].split(",") if metadata["emotions"] else [],
                    "source": metadata["source"] if metadata["source"] else None,
                    "context": eval(metadata["context"]) if metadata["context"] else {},
                    "embedding": results["embeddings"][0][i]
                }
                
                memories.append(MemoryItem(**memory_data))

            return memories

        except Exception as e:
            logger.error(f"Failed to query memories: {e}")
            raise

    async def close(self):
        """Close the database connection"""
        # ChromaDB handles connection cleanup automatically
        pass