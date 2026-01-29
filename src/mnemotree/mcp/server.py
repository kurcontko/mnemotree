from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from mnemotree.core.memory import MemoryCore, ModeDefaultsConfig, NerConfig, RecallFilters
from mnemotree.core.models import MemoryItem, MemoryType, coerce_datetime
from mnemotree.ner import create_ner
from mnemotree.store.protocols import SupportsMemoryListing

_memory_lock = asyncio.Lock()
_memory_core: MemoryCore | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_memory_type(value: str | None) -> MemoryType | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    for memory_type in MemoryType:
        if memory_type.value == normalized or memory_type.name.lower() == normalized:
            return memory_type
    raise ValueError(f"Unknown memory_type '{value}'.")


def _ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _parse_recall_filters(filters: dict[str, Any] | None) -> RecallFilters | None:
    if not filters:
        return None

    raw_memory_types = filters.get("memory_types") or filters.get("memory_type")
    parsed_memory_types: list[MemoryType] | None = None
    if raw_memory_types is not None:
        parsed_list: list[MemoryType] = []
        for raw_value in _ensure_list(raw_memory_types) or []:
            if isinstance(raw_value, MemoryType):
                parsed_list.append(raw_value)
            else:
                parsed = _parse_memory_type(str(raw_value))
                if parsed is not None:
                    parsed_list.append(parsed)
        if parsed_list:
            parsed_memory_types = parsed_list

    tags = _ensure_list(filters.get("tags"))
    if tags is not None:
        tags = [str(tag) for tag in tags]

    return RecallFilters(
        memory_types=parsed_memory_types,
        tags=tags,
        min_importance=filters.get("min_importance"),
        max_importance=filters.get("max_importance"),
        since=filters.get("since"),
        until=filters.get("until"),
        source=filters.get("source"),
        author=filters.get("author"),
        conversation_id=filters.get("conversation_id"),
        user_id=filters.get("user_id"),
    )


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    values = _ensure_list(value) or []
    return [str(tag) for tag in values]


def _serialize_memory(
    memory: MemoryItem,
    *,
    include_embedding: bool,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    data = memory.model_dump(mode="json")
    if fields:
        return {field: data[field] for field in fields if field in data}
    if not include_embedding:
        data.pop("embedding", None)
    return data


def _memory_snippet(memory: MemoryItem, max_len: int = 200) -> str:
    text = memory.summary or memory.content or ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _serialize_memory_index(memory: MemoryItem, rank: int) -> dict[str, Any]:
    data = memory.model_dump(mode="json")
    return {
        "memory_id": memory.memory_id,
        "rank": rank,
        "summary": memory.summary,
        "snippet": _memory_snippet(memory),
        "memory_type": memory.memory_type.value,
        "timestamp": data.get("timestamp"),
        "importance": memory.importance,
        "tags": memory.tags,
    }


def _memory_timestamp(memory: MemoryItem) -> datetime:
    timestamp = coerce_datetime(memory.timestamp, default=None)
    if timestamp is None:
        timestamp = coerce_datetime(memory.last_accessed, default=None)
    if timestamp is None:
        timestamp = datetime.min.replace(tzinfo=timezone.utc)
    return timestamp


async def _get_all_memories(
    memory_core: MemoryCore,
    *,
    include_embeddings: bool,
) -> list[MemoryItem]:
    store = memory_core.store
    if not isinstance(store, SupportsMemoryListing):
        raise NotImplementedError("timeline requires a store that supports list_memories().")
    return await store.list_memories(include_embeddings=include_embeddings)


def _resolve_timeline_anchor(
    sorted_memories: list[MemoryItem],
    *,
    memory_id: str | None,
    timestamp: str | None,
) -> tuple[int | None, str | None]:
    if memory_id:
        for idx, memory in enumerate(sorted_memories):
            if memory.memory_id == memory_id:
                return idx, memory_id
        return None, None

    anchor_time = coerce_datetime(timestamp, default=None)
    if anchor_time is None:
        raise ValueError("Invalid timestamp format.")
    for idx, memory in enumerate(sorted_memories):
        if _memory_timestamp(memory) >= anchor_time:
            return idx, memory.memory_id
    if not sorted_memories:
        return None, None
    return len(sorted_memories) - 1, sorted_memories[-1].memory_id


def _compute_timeline_window(
    anchor_index: int,
    *,
    before: int,
    after: int,
    total: int,
) -> tuple[int, int]:
    window_before = max(0, int(before))
    window_after = max(0, int(after))
    start = max(0, anchor_index - window_before)
    end = min(total, anchor_index + window_after + 1)
    return start, end


def _build_timeline_results(
    slice_memories: list[MemoryItem],
    *,
    start: int,
    anchor_index: int,
    anchor_id: str | None,
    include_anchor: bool,
    include_embedding: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for local_rank, memory in enumerate(slice_memories, start=1):
        idx = start + local_rank - 1
        if not include_anchor and anchor_id and memory.memory_id == anchor_id:
            continue
        entry = _serialize_memory_index(memory, local_rank)
        entry["offset"] = idx - anchor_index
        if anchor_id and memory.memory_id == anchor_id:
            entry["anchor"] = True
        if include_embedding:
            entry["embedding"] = memory.embedding
        results.append(entry)
    return results


async def _get_memory_core() -> MemoryCore:
    global _memory_core
    async with _memory_lock:
        if _memory_core is not None:
            return _memory_core

        persist_dir = os.getenv("MNEMOTREE_MCP_PERSIST_DIR", ".mnemotree/chromadb")
        collection_name = os.getenv("MNEMOTREE_MCP_COLLECTION", "memories")
        chroma_host = os.getenv("MNEMOTREE_MCP_CHROMA_HOST")
        chroma_port = os.getenv("MNEMOTREE_MCP_CHROMA_PORT")
        chroma_ssl = _env_bool("MNEMOTREE_MCP_CHROMA_SSL", False)

        try:
            from mnemotree.store import ChromaMemoryStore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ChromaMemoryStore is required for the MCP server. "
                "Install with `mnemotree[mcp_server]`."
            ) from exc

        if chroma_host and chroma_port:
            store = ChromaMemoryStore(
                host=chroma_host,
                port=int(chroma_port),
                ssl=chroma_ssl,
                collection_name=collection_name,
            )
        else:
            store = ChromaMemoryStore(
                persist_directory=persist_dir,
                collection_name=collection_name,
            )

        await store.initialize()

        ner_backend = os.getenv("MNEMOTREE_MCP_NER_BACKEND")
        ner_model = os.getenv("MNEMOTREE_MCP_NER_MODEL")
        ner = None
        if ner_backend:
            ner_kwargs: dict[str, Any] = {}
            if ner_model:
                if ner_backend.strip().lower() == "gliner":
                    ner_kwargs["model_name"] = ner_model
                else:
                    ner_kwargs["model"] = ner_model
            ner = create_ner(ner_backend, **ner_kwargs)

        enable_ner = _env_bool("MNEMOTREE_MCP_ENABLE_NER", False)
        enable_keywords = _env_bool("MNEMOTREE_MCP_ENABLE_KEYWORDS", False)
        mode_defaults = ModeDefaultsConfig(mode="lite", enable_keywords=enable_keywords)
        ner_config = NerConfig(ner=ner, enable_ner=enable_ner)

        _memory_core = MemoryCore(
            store=store,
            mode_defaults=mode_defaults,
            ner_config=ner_config,
        )
        return _memory_core


async def remember(
    content: str,
    memory_type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    context: dict[str, Any] | None = None,
    analyze: bool | None = None,
    summarize: bool | None = None,
    references: list[str] | None = None,
    memory_id: str | None = None,
    timestamp: str | None = None,
    source: str | None = None,
    author: str | None = None,
    metadata: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    include_embedding: bool = False,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Store a memory entry and return the stored record.

    Args:
        content: The text content to store.
        memory_type: Optional type (semantic, episodic, procedural, declarative).
        importance: Optional score 0.0-1.0.
        tags: Optional list of tags for categorization.
        context: Optional context dictionary.
        analyze: Enable NER analysis if configured.
        summarize: Generate a summary for the memory.
        references: List of memory IDs this memory references.
        memory_id: Custom ID (auto-generated if omitted).
        timestamp: ISO-8601 timestamp (defaults to now).
        source: Source identifier.
        author: Author identifier.
        metadata: Additional metadata dictionary.
        conversation_id: Conversation identifier.
        user_id: User identifier.
        include_embedding: Include embedding vector in response.
        fields: If provided, return only these fields (overrides include_embedding).

    Returns:
        The stored memory record as a dictionary.
    """
    memory_core = await _get_memory_core()
    parsed_type = _parse_memory_type(memory_type)
    remember_kwargs: dict[str, Any] = {
        "content": content,
        "memory_type": parsed_type,
        "importance": importance,
        "tags": tags,
        "context": context,
        "analyze": analyze,
        "summarize": summarize,
        "references": references,
    }
    if memory_id is not None:
        remember_kwargs["memory_id"] = memory_id
    if timestamp is not None:
        remember_kwargs["timestamp"] = timestamp
    if source is not None:
        remember_kwargs["source"] = source
    if author is not None:
        remember_kwargs["author"] = author
    if metadata is not None:
        remember_kwargs["metadata"] = metadata
    if conversation_id is not None:
        remember_kwargs["conversation_id"] = conversation_id
    if user_id is not None:
        remember_kwargs["user_id"] = user_id
    memory = await memory_core.remember(**remember_kwargs)
    return _serialize_memory(memory, include_embedding=include_embedding, fields=fields)


async def recall(
    query: str,
    limit: int = 10,
    scoring: bool = True,
    update_access: bool = False,
    compact: bool = False,
    include_summary: bool = True,
    include_embedding: bool = False,
    fields: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve memories relevant to a query string.

    Args:
        query: Search query text.
        limit: Maximum results to return (default: 10).
        scoring: Enable relevance scoring (default: True).
        update_access: Update last_accessed timestamp (default: False).
        compact: Return compact ranked results (default: False).
        include_summary: Include summary in compact results (default: True).
        include_embedding: Include embedding vectors in results.
        fields: If provided, return only these fields (overrides include_embedding).
        filters: Optional dict with keys:
            - memory_types: List of types to include.
            - tags: List of tags to filter by.
            - min_importance / max_importance: Float thresholds.
            - since / until: ISO-8601 timestamps for time range.
            - source, author, conversation_id, user_id: String filters.
        candidate_limit: Fetch more candidates before filtering (improves recall).

    Returns:
        List of matching memory dictionaries.
    """
    memory_core = await _get_memory_core()
    if compact and fields is not None:
        raise ValueError("fields is not supported when compact=True.")
    parsed_filters = _parse_recall_filters(filters)
    recall_kwargs: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "scoring": scoring,
        "update_access": update_access,
    }
    if parsed_filters is not None:
        recall_kwargs["filters"] = parsed_filters
    if candidate_limit is not None:
        recall_kwargs["candidate_limit"] = candidate_limit
    memories = await memory_core.recall(**recall_kwargs)
    if compact:
        results: list[dict[str, Any]] = []
        for rank, memory in enumerate(memories, start=1):
            item = _serialize_memory_index(memory, rank)
            if not include_summary:
                item.pop("summary", None)
            if include_embedding:
                item["embedding"] = memory.embedding
            results.append(item)
        return results
    return [
        _serialize_memory(memory, include_embedding=include_embedding, fields=fields)
        for memory in memories
    ]


async def timeline(
    memory_id: str | None = None,
    timestamp: str | None = None,
    before: int = 3,
    after: int = 3,
    include_anchor: bool = True,
    include_embedding: bool = False,
) -> list[dict[str, Any]]:
    """Return memories around a given memory or timestamp in chronological order.

    Args:
        memory_id: Anchor memory ID (provide this or timestamp).
        timestamp: ISO-8601 anchor timestamp (provide this or memory_id).
        before: Number of memories before anchor (default: 3).
        after: Number of memories after anchor (default: 3).
        include_anchor: Include anchor memory in results (default: True).
        include_embedding: Include embedding vectors (default: False).

    Returns:
        List of memories sorted chronologically with offset from anchor.
    """
    if not memory_id and not timestamp:
        raise ValueError("Provide either memory_id or timestamp.")
    memory_core = await _get_memory_core()
    memories = await _get_all_memories(memory_core, include_embeddings=include_embedding)
    if not memories:
        return []

    sorted_memories = sorted(memories, key=_memory_timestamp)
    anchor_index, anchor_id = _resolve_timeline_anchor(
        sorted_memories, memory_id=memory_id, timestamp=timestamp
    )
    if anchor_index is None:
        return []

    start, end = _compute_timeline_window(
        anchor_index, before=before, after=after, total=len(sorted_memories)
    )
    slice_memories = sorted_memories[start:end]
    return _build_timeline_results(
        slice_memories,
        start=start,
        anchor_index=anchor_index,
        anchor_id=anchor_id,
        include_anchor=include_anchor,
        include_embedding=include_embedding,
    )


async def get_memories(
    memory_ids: list[str],
    include_embedding: bool = False,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch full memory records by ID.

    Args:
        memory_ids: List of memory IDs to retrieve.
        include_embedding: Include embedding vectors (default: False).
        fields: If provided, return only these fields in each result.

    Returns:
        List of memory dictionaries (missing IDs are omitted).
    """
    memory_core = await _get_memory_core()
    store = memory_core.store
    if not memory_ids:
        return []
    results = await asyncio.gather(*(store.get_memory(memory_id) for memory_id in memory_ids))
    return [
        _serialize_memory(memory, include_embedding=include_embedding, fields=fields)
        for memory in results
        if memory is not None
    ]


async def update_memory(
    memory_id: str,
    patch: dict[str, Any],
    *,
    reembed: bool = False,
    include_embedding: bool = False,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Update fields on an existing memory.

    Args:
        memory_id: ID of the memory to update.
        patch: Dict of fields to update. Supported keys:
            content, summary, tags, importance, context, metadata, source, author,
            timestamp, memory_type, conversation_id, user_id.
        reembed: Recompute embedding when content changes (default: False).
        include_embedding: Include embedding vector in response.
        fields: If provided, return only these fields (overrides include_embedding).

    Returns:
        Updated memory record as a dictionary.
    """
    if not patch:
        raise ValueError("patch is required.")
    allowed_fields = {
        "content",
        "summary",
        "tags",
        "importance",
        "context",
        "metadata",
        "source",
        "author",
        "timestamp",
        "memory_type",
        "conversation_id",
        "user_id",
    }
    unknown_fields = {str(key) for key in patch} - allowed_fields
    if unknown_fields:
        raise ValueError(f"Unknown update fields: {', '.join(sorted(unknown_fields))}.")

    memory_core = await _get_memory_core()
    store = memory_core.store
    memory = await store.get_memory(memory_id)
    if memory is None:
        raise ValueError("Memory not found.")

    content_updated = False

    if "content" in patch:
        memory.content = str(patch["content"])
        content_updated = True
    if "summary" in patch:
        memory.summary = patch["summary"]
    if "tags" in patch:
        memory.tags = _coerce_tags(patch["tags"])
    if "importance" in patch:
        importance = patch["importance"]
        if importance is None:
            raise ValueError("importance cannot be null.")
        importance = float(importance)
        if not 0 <= importance <= 1:
            raise ValueError("importance must be between 0 and 1")
        memory.importance = importance
    if "context" in patch:
        memory.context = patch["context"]
    if "metadata" in patch:
        metadata = patch["metadata"]
        if metadata is None:
            memory.metadata = {}
        elif not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        else:
            memory.metadata.update(metadata)
    if "source" in patch:
        memory.source = patch["source"]
    if "author" in patch:
        memory.author = patch["author"]
    if "timestamp" in patch:
        parsed_timestamp = coerce_datetime(patch["timestamp"], default=None)
        if parsed_timestamp is None:
            raise ValueError("Invalid timestamp format.")
        memory.timestamp = parsed_timestamp
    if "memory_type" in patch:
        raw_type = patch["memory_type"]
        if raw_type is None:
            raise ValueError("memory_type cannot be null.")
        if isinstance(raw_type, MemoryType):
            memory.memory_type = raw_type
        else:
            parsed_type = _parse_memory_type(str(raw_type))
            if parsed_type is None:
                raise ValueError("memory_type cannot be null.")
            memory.memory_type = parsed_type
    if "conversation_id" in patch:
        memory.conversation_id = patch["conversation_id"]
    if "user_id" in patch:
        memory.user_id = patch["user_id"]

    if content_updated and reembed:
        memory.embedding = await memory_core.get_embedding(memory.content)

    await store.store_memory(memory)
    return _serialize_memory(memory, include_embedding=include_embedding, fields=fields)


async def forget(memory_id: str) -> bool:
    """Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        True if deleted, False if not found.
    """
    memory_core = await _get_memory_core()
    return await memory_core.forget(memory_id)


async def reflect(min_importance: float = 0.7) -> dict[str, Any]:
    """Summarize patterns across higher-importance memories.

    Args:
        min_importance: Only include memories with importance >= this value (default: 0.7).

    Returns:
        Dictionary with pattern summary and insights.
    """
    memory_core = await _get_memory_core()
    return await memory_core.reflect(min_importance=min_importance)


def memory_types() -> list[str]:
    """Return supported memory type values.

    Returns:
        List of valid memory_type strings: semantic, episodic, procedural, declarative.
    """
    return [memory_type.value for memory_type in MemoryType]


def health() -> dict[str, str]:
    """Simple health check.

    Returns:
        Dictionary with status key ("ok" if healthy).
    """
    return {"status": "ok"}


_FASTMCP_IMPORT_ERROR = (
    "FastMCP is required to run the MCP server. Install with `mnemotree[mcp_server]`."
)
_mcp_instance: Any | None = None


def _register_tools(mcp: Any) -> None:
    mcp.tool(remember)
    mcp.tool(recall)
    mcp.tool(timeline)
    mcp.tool(get_memories)
    mcp.tool(update_memory)
    mcp.tool(forget)
    mcp.tool(reflect)
    mcp.tool(memory_types)
    mcp.tool(health)


def _load_fastmcp() -> Any:
    try:
        from fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_FASTMCP_IMPORT_ERROR) from exc
    return FastMCP


def _get_mcp() -> Any:
    global _mcp_instance
    if _mcp_instance is None:
        fastmcp_cls = _load_fastmcp()
        mcp_instance = fastmcp_cls("Mnemotree Memory")
        _register_tools(mcp_instance)
        _mcp_instance = mcp_instance
    return _mcp_instance


class _LazyMCP:
    def __getattr__(self, name: str) -> Any:
        return getattr(_get_mcp(), name)

    def __dir__(self) -> list[str]:
        return dir(_get_mcp())


mcp = _LazyMCP()


def main() -> None:
    """Entry point for the MCP server CLI."""
    _get_mcp().run()


if __name__ == "__main__":
    main()
