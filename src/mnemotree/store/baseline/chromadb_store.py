from __future__ import annotations

from typing import List, Optional, Dict, Any
import numpy as np
from neo4j import AsyncGraphDatabase
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
import logging

from ...core.models import MemoryItem, MemoryType
from ...core.query import MemoryQuery
from ..base import BaseMemoryStore


logger = logging.getLogger(__name__)


class BaselineChromaStore(BaseMemoryStore):
    """ChromaDB baseline using only embeddings"""
    
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
        """Initialize collection"""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    async def store_memory(self, memory: MemoryItem) -> None:
        """Store memory with only embeddings"""
        self.collection.add(
            ids=[memory.memory_id],
            embeddings=[memory.embedding],
            documents=[memory.content],
            metadatas=[{
                "memory_id": memory.memory_id
            }]
        )
        
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory by ID"""
        result = self.collection.get(
            ids=[memory_id],
            include=["embeddings", "documents"]
        )
        
        if not result["ids"]:
            return None
            
        return MemoryItem(
            memory_id=memory_id,
            content=result["documents"][0],
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            embedding=result["embeddings"][0]
        )
        
    async def get_similar_memories(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Get similar memories using only vector similarity"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["embeddings", "documents", "metadatas"]
        )
        
        memories = []
        for i, memory_id in enumerate(results["ids"][0]):
            memories.append(MemoryItem(
                memory_id=memory_id,
                content=results["documents"][0][i],
                memory_type=MemoryType.SEMANTIC,
                importance=0.5,
                embedding=results["embeddings"][0][i]
            ))
            
        return memories
    
    async def update_connections(
        self,
        memory_id: str,
        related_ids: Optional[List[str]] = None,
        conflict_ids: Optional[List[str]] = None,
        previous_id: Optional[str] = None,
        next_id: Optional[str] = None
    ) -> None:
        """Update memory connections"""
        raise NotImplementedError("Connections are not supported in ChromaDB")

    async def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Execute a complex memory query"""
        raise NotImplementedError("Query operations are not supported in ChromaDB")
    
    async def query_by_entities(self, entities: Dict[str, str]) -> List[MemoryItem]:
        """Query memories by entities"""
        raise NotImplementedError("Entity queries are not supported in ChromaDB")
