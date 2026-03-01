from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from ..core.models import LinkType, MemoryItem, MemoryLink
from ..core.query import MemoryQuery


@runtime_checkable
class SupportsClose(Protocol):
    async def close(self) -> None: ...


@runtime_checkable
class MemoryCRUDStore(SupportsClose, Protocol):
    async def store_memory(self, memory: MemoryItem) -> None: ...

    async def get_memory(self, memory_id: str) -> MemoryItem | None: ...

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool: ...


@runtime_checkable
class SupportsMetadataUpdate(Protocol):
    async def update_memory_metadata(self, memory_id: str, metadata: dict[str, Any]) -> bool: ...


@runtime_checkable
class SupportsConnections(Protocol):
    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: list[str] | None = None,
        conflict_ids: list[str] | None = None,
        previous_id: str | None = None,
        next_id: str | None = None,
    ) -> None: ...


@runtime_checkable
class SupportsVectorSearch(Protocol):
    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsStructuredQuery(Protocol):
    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsEntityQuery(Protocol):
    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsMemoryListing(Protocol):
    async def list_memories(
        self,
        *,
        include_embeddings: bool = False,
    ) -> list[MemoryItem]: ...


@runtime_checkable
class SupportsKnowledgeGraph(Protocol):
    """Protocol for knowledge graph operations supporting Zettelkasten-style linking."""

    async def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        *,
        strength: float = 1.0,
        context: str | None = None,
        bidirectional: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryLink:
        """Create a directed link between two memories.

        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory
            link_type: Type of semantic relationship
            strength: Initial link strength (0-1)
            context: Optional explanation of why this link exists
            bidirectional: If True, create reverse link as well
            metadata: Additional metadata for the link

        Returns:
            The created MemoryLink object
        """
        ...

    async def get_links(
        self,
        memory_id: str,
        *,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        link_types: list[LinkType] | None = None,
        min_strength: float = 0.0,
    ) -> list[MemoryLink]:
        """Get all links for a memory.

        Args:
            memory_id: ID of the memory
            direction: Which links to retrieve (outgoing/incoming/both)
            link_types: Optional filter for specific link types
            min_strength: Minimum link strength threshold

        Returns:
            List of MemoryLink objects
        """
        ...

    async def get_backlinks(
        self,
        memory_id: str,
        *,
        link_types: list[LinkType] | None = None,
    ) -> list[MemoryItem]:
        """Get all memories that link TO this memory (backlinks).

        Args:
            memory_id: ID of the target memory
            link_types: Optional filter for specific link types

        Returns:
            List of MemoryItem objects that link to this memory
        """
        ...

    async def traverse_graph(
        self,
        start_id: str,
        *,
        max_depth: int = 3,
        link_types: list[LinkType] | None = None,
        strategy: Literal["bfs", "dfs"] = "bfs",
    ) -> list[tuple[MemoryItem, int, list[MemoryLink]]]:
        """Traverse the knowledge graph from a starting memory.

        Args:
            start_id: ID of the starting memory
            max_depth: Maximum traversal depth (default 3, max 5)
            link_types: Optional filter for specific link types
            strategy: Traversal strategy (breadth-first or depth-first)

        Returns:
            List of tuples: (memory, depth, path_links)
        """
        ...

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 5,
    ) -> list[tuple[MemoryItem, MemoryLink]] | None:
        """Find shortest path between two memories.

        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory
            max_depth: Maximum path length to search

        Returns:
            List of (memory, link) tuples representing the path, or None if no path exists
        """
        ...

    async def suggest_links(
        self,
        memory_id: str,
        *,
        limit: int = 10,
        min_similarity: float = 0.7,
        use_llm: bool = False,
    ) -> list[tuple[MemoryItem, LinkType, float, str | None]]:
        """Suggest potential links for a memory.

        Args:
            memory_id: ID of the memory to find links for
            limit: Maximum number of suggestions
            min_similarity: Minimum similarity threshold
            use_llm: Whether to use LLM for refining suggestions (pro mode)

        Returns:
            List of tuples: (target_memory, suggested_link_type, confidence, reasoning)
        """
        ...

    async def update_link_strength(
        self,
        link_id: str,
        strength: float,
    ) -> bool:
        """Update the strength of an existing link.

        Args:
            link_id: ID of the link to update
            strength: New strength value (0-1)

        Returns:
            True if successful, False if link not found
        """
        ...

    async def delete_link(
        self,
        link_id: str,
    ) -> bool:
        """Delete a link.

        Args:
            link_id: ID of the link to delete

        Returns:
            True if successful, False if link not found
        """
        ...

    async def get_neighborhood_links(
        self,
        memory_ids: list[str],
        max_depth: int = 3,
        link_types: list[LinkType] | None = None,
    ) -> list[MemoryLink]:
        """Return all links reachable within max_depth hops from seed ids.

        Used by PPRGraphRetriever to build the adjacency matrix for Personalized
        PageRank without loading the full graph.

        Args:
            memory_ids: Seed memory IDs to expand from.
            max_depth: Maximum hop depth (capped internally).
            link_types: Optional whitelist of link types to traverse.

        Returns:
            Flat list of MemoryLink objects (may include duplicates across seeds).
        """
        ...

    async def traverse_typed_path(
        self,
        start_id: str,
        path_pattern: list[LinkType],
        max_results: int = 20,
    ) -> list[tuple[MemoryItem, list[MemoryLink]]]:
        """Traverse a specific typed sequence of link types from a start memory.

        Implements Semantic GraphRAG-style typed-path queries, where each hop
        is constrained to a particular link type.

        Args:
            start_id: ID of the starting memory.
            path_pattern: Ordered list of LinkType values to traverse.
                e.g. [LinkType.CAUSES, LinkType.PART_OF] means two hops:
                first only CAUSES links, then only PART_OF links.
            max_results: Maximum number of end-node results to return.

        Returns:
            List of (end_memory, path_links) tuples where path_links is the
            sequence of MemoryLink objects traversed to reach end_memory.
        """
        ...
