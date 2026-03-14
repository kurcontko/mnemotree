from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from mnemotree.core.memory import (
    MemoryCore,
    ModeDefaultsConfig,
    NerConfig,
    RecallFilters,
    RetrievalConfig,
)
from mnemotree.core.models import LinkType, MemoryItem, MemoryLink, MemoryType, coerce_datetime
from mnemotree.ner import create_ner
from mnemotree.store.base import BaseMemoryStore
from mnemotree.store.protocols import SupportsKnowledgeGraph, SupportsMemoryListing

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
    result_rank = 0
    for i, memory in enumerate(slice_memories):
        idx = start + i
        if not include_anchor and anchor_id and memory.memory_id == anchor_id:
            continue
        result_rank += 1
        entry = _serialize_memory_index(memory, result_rank)
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

        persist_dir = os.getenv("MNEMOTREE_MCP_PERSIST_DIR", ".mnemotree/mnemotree.sqlite")
        collection_name = os.getenv("MNEMOTREE_MCP_COLLECTION", "memories")

        # Support legacy ChromaDB via env vars, otherwise default to SQLite (zero-infra).
        chroma_host = os.getenv("MNEMOTREE_MCP_CHROMA_HOST")
        chroma_port = os.getenv("MNEMOTREE_MCP_CHROMA_PORT")
        store_backend = os.getenv("MNEMOTREE_MCP_STORE_BACKEND", "sqlite")

        store: BaseMemoryStore
        if chroma_host and chroma_port or store_backend.lower() == "chroma":
            try:
                from mnemotree.store import ChromaMemoryStore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "ChromaMemoryStore requires optional dependencies. "
                    "Install with `pip install mnemotree[chroma]`."
                ) from exc

            chroma_ssl = _env_bool("MNEMOTREE_MCP_CHROMA_SSL", False)
            if chroma_host and chroma_port:
                store = ChromaMemoryStore(
                    host=chroma_host,
                    port=int(chroma_port),
                    ssl=chroma_ssl,
                    collection_name=collection_name,
                )
            else:
                chroma_dir = os.getenv("MNEMOTREE_MCP_PERSIST_DIR", ".mnemotree/chromadb")
                store = ChromaMemoryStore(
                    persist_directory=chroma_dir,
                    collection_name=collection_name,
                )
        else:
            from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

            store = SQLiteVecMemoryStore(
                db_path=persist_dir,
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
        enable_bm25 = _env_bool("MNEMOTREE_MCP_ENABLE_BM25", True)
        mode_defaults = ModeDefaultsConfig(mode="lite", enable_keywords=enable_keywords)
        ner_config = NerConfig(ner=ner, enable_ner=enable_ner)
        retrieval_config = RetrievalConfig(
            retrieval_mode="hybrid",
            enable_bm25=enable_bm25,
        )

        _memory_core = MemoryCore(
            store=store,
            mode_defaults=mode_defaults,
            ner_config=ner_config,
            retrieval_config=retrieval_config,
        )
        return _memory_core


async def remember(
    content: str,
    memory_type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a memory entry and return the stored record.

    Args:
        content: The text content to store.
        memory_type: Optional type (semantic, episodic, procedural, declarative).
        importance: Optional score 0.0-1.0.
        tags: Optional list of tags for categorization.
        context: Optional context dictionary.
        metadata: Additional metadata dictionary.

    Returns:
        The stored memory record as a dictionary.
    """
    if not content or not content.strip():
        raise ValueError("content cannot be empty")
    if importance is not None and not (0.0 <= importance <= 1.0):
        raise ValueError("importance must be between 0.0 and 1.0")
    memory_core = await _get_memory_core()
    parsed_type = _parse_memory_type(memory_type)
    remember_kwargs: dict[str, Any] = {
        "content": content,
        "memory_type": parsed_type,
        "importance": importance,
        "tags": tags,
        "context": context,
    }
    if metadata is not None:
        remember_kwargs["metadata"] = metadata
    memory = await memory_core.remember(**remember_kwargs)
    return _serialize_memory(memory, include_embedding=False)


async def recall(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    compact: bool = True,
    include_summary: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve memories relevant to a query string.

    Args:
        query: Search query text.
        limit: Maximum results to return (default: 10).
        filters: Optional dict with keys:
            - memory_types: List of types to include.
            - tags: List of tags to filter by.
            - min_importance / max_importance: Float thresholds.
            - since / until: ISO-8601 timestamps for time range.
            - source, author, conversation_id, user_id: String filters.
        compact: Return compact ranked results (default: True).
        include_summary: Include summary in compact results (default: True).

    Returns:
        List of matching memory dictionaries.
    """
    memory_core = await _get_memory_core()
    parsed_filters = _parse_recall_filters(filters)
    recall_kwargs: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "scoring": True,
        "update_access": False,
    }
    if parsed_filters is not None:
        recall_kwargs["filters"] = parsed_filters
    memories = await memory_core.recall(**recall_kwargs)
    if compact:
        results: list[dict[str, Any]] = []
        for rank, memory in enumerate(memories, start=1):
            item = _serialize_memory_index(memory, rank)
            if not include_summary:
                item.pop("summary", None)
            results.append(item)
        return results
    return [_serialize_memory(memory, include_embedding=False) for memory in memories]


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
) -> list[dict[str, Any]]:
    """Fetch full memory records by ID.

    Args:
        memory_ids: List of memory IDs to retrieve.

    Returns:
        List of memory dictionaries (missing IDs are omitted).
    """
    memory_core = await _get_memory_core()
    store = memory_core.store
    if not memory_ids:
        return []
    results = await asyncio.gather(*(store.get_memory(memory_id) for memory_id in memory_ids))
    return [
        _serialize_memory(memory, include_embedding=False)
        for memory in results
        if memory is not None
    ]


async def update_memory(
    memory_id: str,
    patch: dict[str, Any],
    *,
    reembed: bool = False,
) -> dict[str, Any]:
    """Update fields on an existing memory.

    Args:
        memory_id: ID of the memory to update.
        patch: Dict of fields to update. Supported keys:
            content, summary, tags, importance, context, metadata,
            is_hot, event_time, valid_from, valid_until, contextual_intent.
        reembed: Recompute embedding when content changes (default: False).

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
        "is_hot",
        "event_time",
        "valid_from",
        "valid_until",
        "contextual_intent",
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
        raw = patch["content"]
        if raw is None:
            raise ValueError("content cannot be null.")
        content_str = str(raw).strip()
        if not content_str:
            raise ValueError("content cannot be empty.")
        memory.content = content_str
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
    if "is_hot" in patch:
        memory.is_hot = bool(patch["is_hot"])
    if "contextual_intent" in patch:
        memory.contextual_intent = patch["contextual_intent"] or None
    for dt_field in ("event_time", "valid_from", "valid_until"):
        if dt_field in patch:
            val = patch[dt_field]
            setattr(memory, dt_field, coerce_datetime(val, default=None) if val else None)

    if content_updated and reembed:
        memory.embedding = await memory_core.get_embedding(memory.content)

    await store.store_memory(memory)
    return _serialize_memory(memory, include_embedding=False)


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


def _parse_link_type(value: str | None) -> LinkType:
    if value is None:
        return LinkType.REFERENCES
    normalized = value.strip().lower()
    for lt in LinkType:
        if lt.value == normalized or lt.name.lower() == normalized:
            return lt
    raise ValueError(f"Unknown link_type '{value}'. Valid: {[lt.value for lt in LinkType]}")


def _serialize_link(link: MemoryLink) -> dict[str, Any]:
    return {
        "link_id": link.link_id,
        "source_id": link.source_id,
        "target_id": link.target_id,
        "link_type": link.link_type.value,
        "strength": link.strength,
        "context": link.context,
        "created_at": link.created_at.isoformat() if link.created_at else None,
    }


async def link_memories(
    source_id: str,
    target_id: str,
    link_type: str | None = None,
    context: str | None = None,
    bidirectional: bool = False,
) -> dict[str, Any]:
    """Create a typed link between two memories.

    Args:
        source_id: ID of the source memory.
        target_id: ID of the target memory.
        link_type: Relationship type (references, elaborates, contradicts,
            supports, derives_from, similar_to, supersedes, updates, sequence).
            Default: references.
        context: Optional explanation of the relationship.
        bidirectional: Create reverse link as well (default: False).

    Returns:
        The created link record.
    """
    memory_core = await _get_memory_core()
    parsed_type = _parse_link_type(link_type)
    link = await memory_core.link(
        source_id,
        target_id,
        parsed_type,
        context=context,
        bidirectional=bidirectional,
    )
    return _serialize_link(link)


async def get_links(
    memory_id: str,
    direction: str = "both",
    link_types: list[str] | None = None,
    min_strength: float = 0.0,
) -> list[dict[str, Any]]:
    """Get all links for a memory.

    Args:
        memory_id: ID of the memory.
        direction: Which links to retrieve (outgoing, incoming, both). Default: both.
        link_types: Optional list of link type names to filter by.
        min_strength: Minimum link strength threshold (default: 0.0).

    Returns:
        List of link records.
    """
    memory_core = await _get_memory_core()
    parsed_types = [_parse_link_type(lt) for lt in link_types] if link_types else None
    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(f"direction must be 'outgoing', 'incoming', or 'both', got '{direction}'")
    links = await memory_core.get_links(
        memory_id,
        direction=direction,  # type: ignore[arg-type]
        link_types=parsed_types,
        min_strength=min_strength,
    )
    return [_serialize_link(link) for link in links]


async def traverse_graph(
    memory_id: str,
    depth: int = 2,
) -> dict[str, Any]:
    """Explore the knowledge graph around a memory.

    Args:
        memory_id: Starting memory ID.
        depth: How many hops to explore (default: 2, max: 5).

    Returns:
        Graph structure with nodes, edges, and counts.
    """
    memory_core = await _get_memory_core()
    return await memory_core.explore(memory_id, depth=min(depth, 5))


async def suggest_links(
    memory_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Suggest potential links for a memory based on semantic similarity.

    Args:
        memory_id: ID of the memory to find suggestions for.
        limit: Maximum number of suggestions (default: 5).

    Returns:
        List of suggested link records with target memory info and scores.
    """
    memory_core = await _get_memory_core()
    store = memory_core.store
    if not isinstance(store, SupportsKnowledgeGraph):
        raise RuntimeError("Store does not support knowledge graph operations.")
    suggestions = await store.suggest_links(memory_id, limit=limit)
    results: list[dict[str, Any]] = []
    for suggestion in suggestions:
        entry: dict[str, Any] = {}
        if isinstance(suggestion, dict):
            entry = suggestion
        elif isinstance(suggestion, tuple) and len(suggestion) >= 2:
            # suggest_links returns (MemoryItem, LinkType, score, reason|None)
            target_memory = suggestion[0]
            link_type = suggestion[1] if len(suggestion) > 1 else None
            score = suggestion[2] if len(suggestion) > 2 else 0.0
            reason = suggestion[3] if len(suggestion) > 3 else None
            entry = {
                "target_id": target_memory.memory_id
                if isinstance(target_memory, MemoryItem)
                else str(target_memory),
                "score": float(score),
            }
            if link_type is not None:
                entry["link_type"] = (
                    link_type.value if hasattr(link_type, "value") else str(link_type)
                )
            if reason:
                entry["reason"] = reason
            if isinstance(target_memory, MemoryItem):
                entry["snippet"] = _memory_snippet(target_memory)
                entry["memory_type"] = target_memory.memory_type.value
        else:
            entry = {"raw": str(suggestion)}
        results.append(entry)
    return results


async def get_conflicts(
    memory_id: str,
) -> dict[str, Any]:
    """Get conflicts for a memory (memories it contradicts).

    Args:
        memory_id: ID of the memory to check.

    Returns:
        Dictionary with the memory's conflict info and related conflicting memories.
    """
    memory_core = await _get_memory_core()
    memory = await memory_core.store.get_memory(memory_id)
    if memory is None:
        raise ValueError(f"Memory '{memory_id}' not found.")

    result: dict[str, Any] = {
        "memory_id": memory_id,
        "snippet": _memory_snippet(memory),
        "conflicts_with": memory.conflicts_with or [],
        "conflicting_memories": [],
    }

    if memory.conflicts_with:
        conflict_tasks = [memory_core.store.get_memory(cid) for cid in memory.conflicts_with]
        conflict_memories = await asyncio.gather(*conflict_tasks)
        for cm in conflict_memories:
            if cm is not None:
                result["conflicting_memories"].append(
                    {
                        "memory_id": cm.memory_id,
                        "snippet": _memory_snippet(cm),
                        "importance": cm.importance,
                        "memory_type": cm.memory_type.value,
                    }
                )

    # Also check for CONTRADICTS links
    try:
        links = await memory_core.get_links(
            memory_id,
            link_types=[LinkType.CONTRADICTS],
        )
        result["contradiction_links"] = [_serialize_link(link) for link in links]
    except RuntimeError:
        result["contradiction_links"] = []

    return result


async def resolve_conflict(
    memory_id: str,
    conflicting_id: str,
    resolution: str = "keep_both",
    winner_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a conflict between two memories.

    Args:
        memory_id: ID of the first memory.
        conflicting_id: ID of the conflicting memory.
        resolution: Resolution strategy:
            - keep_both: Keep both, remove from conflicts list.
            - keep_newer: Delete the older memory.
            - keep_winner: Delete the loser (requires winner_id).
            - supersede: Mark winner as superseding loser with SUPERSEDES link.
        winner_id: Required when resolution is 'keep_winner' or 'supersede'.

    Returns:
        Dictionary describing the resolution action taken.
    """
    memory_core = await _get_memory_core()
    m1 = await memory_core.store.get_memory(memory_id)
    m2 = await memory_core.store.get_memory(conflicting_id)
    if m1 is None:
        raise ValueError(f"Memory '{memory_id}' not found.")
    if m2 is None:
        raise ValueError(f"Memory '{conflicting_id}' not found.")

    result: dict[str, Any] = {
        "memory_id": memory_id,
        "conflicting_id": conflicting_id,
        "resolution": resolution,
    }

    if resolution == "keep_both":
        # Remove from each other's conflicts_with
        m1.conflicts_with = [c for c in (m1.conflicts_with or []) if c != conflicting_id]
        m2.conflicts_with = [c for c in (m2.conflicts_with or []) if c != memory_id]
        await memory_core.store.store_memory(m1)
        await memory_core.store.store_memory(m2)
        result["action"] = "Removed from conflicts lists, both memories kept."

    elif resolution == "keep_newer":
        t1 = coerce_datetime(m1.timestamp, default=None)
        t2 = coerce_datetime(m2.timestamp, default=None)
        if t1 is None or t2 is None:
            raise ValueError(
                "Cannot resolve 'keep_newer': one or both memories have unparseable timestamps."
            )
        if t1 >= t2:
            await memory_core.forget(conflicting_id, cascade=True)
            result["action"] = f"Deleted older memory {conflicting_id}."
            result["deleted"] = conflicting_id
            result["kept"] = memory_id
        else:
            await memory_core.forget(memory_id, cascade=True)
            result["action"] = f"Deleted older memory {memory_id}."
            result["deleted"] = memory_id
            result["kept"] = conflicting_id

    elif resolution == "keep_winner":
        if not winner_id or winner_id not in (memory_id, conflicting_id):
            raise ValueError("winner_id must be one of memory_id or conflicting_id.")
        loser_id = conflicting_id if winner_id == memory_id else memory_id
        await memory_core.forget(loser_id, cascade=True)
        result["action"] = f"Deleted loser memory {loser_id}."
        result["deleted"] = loser_id
        result["kept"] = winner_id

    elif resolution == "supersede":
        if not winner_id or winner_id not in (memory_id, conflicting_id):
            raise ValueError("winner_id must be one of memory_id or conflicting_id.")
        loser_id = conflicting_id if winner_id == memory_id else memory_id
        try:
            link = await memory_core.link(
                winner_id,
                loser_id,
                LinkType.SUPERSEDES,
                context="Conflict resolution: superseded",
            )
            result["link"] = _serialize_link(link)
        except RuntimeError as exc:
            result["link_error"] = str(exc)
        result["action"] = f"Memory {winner_id} now supersedes {loser_id}."
        result["winner"] = winner_id
        result["loser"] = loser_id

    else:
        raise ValueError(
            f"Unknown resolution '{resolution}'. "
            "Valid: keep_both, keep_newer, keep_winner, supersede."
        )

    return result


async def consolidate(
    user_id: str | None = None,
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.7,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run offline memory consolidation (RAPTOR-style episodic→semantic promotion).

    Clusters similar episodic memories, generates semantic summaries, and
    optionally deprecates low-signal memories. Requires an LLM to be configured.

    Args:
        user_id: Scope consolidation to this user's memories (default: all).
        min_cluster_size: Minimum memories per cluster (default: 3).
        similarity_threshold: Clustering similarity threshold (default: 0.7).
        dry_run: If True, report what would happen without making changes (default: False).

    Returns:
        Dictionary with consolidation statistics.
    """
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    memory_core = await _get_memory_core()
    from mnemotree.experimental.consolidation import ConsolidationConfig

    config = ConsolidationConfig(
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
        dry_run=dry_run,
    )
    result = await memory_core.consolidate(user_id=user_id, config=config)
    return {
        "total_memories_processed": result.total_memories_processed,
        "clusters_formed": result.clusters_formed,
        "semantic_memories_created": result.semantic_memories_created,
        "memories_deprecated": result.memories_deprecated,
        "created_semantic_ids": result.created_semantic_ids,
        "deprecated_memory_ids": result.deprecated_memory_ids,
        "duration_seconds": result.duration_seconds,
        "dry_run": dry_run,
    }


async def judge_conflicts(
    memory_ids: list[str] | None = None,
    query: str | None = None,
    similarity_threshold: float = 0.92,
    limit: int = 20,
) -> dict[str, Any]:
    """Detect conflicts among memories (AMA Judge pattern).

    Provide either memory_ids or a query to select memories to check.
    Uses fast cosine-threshold detection — no LLM required.

    Args:
        memory_ids: Specific memory IDs to check pairwise.
        query: Recall query to find memories to check.
        similarity_threshold: Cosine similarity threshold (default: 0.92).
        limit: Max memories to check when using query (default: 20).

    Returns:
        Dictionary with detected conflicts.
    """
    memory_core = await _get_memory_core()
    memories: list[MemoryItem] = []

    if memory_ids:
        results = await asyncio.gather(*(memory_core.store.get_memory(mid) for mid in memory_ids))
        memories = [m for m in results if m is not None]
    elif query:
        memories = await memory_core.recall(query, limit=limit)
    else:
        raise ValueError("Provide either memory_ids or query.")

    conflicts = memory_core.judge_conflicts(memories, similarity_threshold=similarity_threshold)
    return {
        "memories_checked": len(memories),
        "conflicts_found": len(conflicts),
        "conflicts": conflicts,
    }


async def observe(
    content: str,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Extract and store observations from a conversation turn (Mastra OM pattern).

    The observer extracts memorable facts from the content and stores them
    as episodic memories. No per-turn retrieval is performed — write-only
    for minimal latency.

    Args:
        content: The conversation turn text to observe.
        user_id: Optional user ID for stored memories.
        conversation_id: Optional conversation ID.

    Returns:
        Dictionary with observation results including stored memory IDs.
    """
    memory_core = await _get_memory_core()
    memory_ids = await memory_core.observe(
        content,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    return {
        "observations_stored": len(memory_ids),
        "memory_ids": memory_ids,
    }


def link_types() -> list[str]:
    """Return all supported link type values.

    Returns:
        List of valid link_type strings.
    """
    return [lt.value for lt in LinkType]


_FASTMCP_IMPORT_ERROR = (
    "FastMCP is required to run the MCP server. Install with `mnemotree[mcp_server]`."
)
_mcp_instance: Any | None = None


def _register_tools(mcp: Any) -> None:
    mcp.tool(remember)
    mcp.tool(recall)
    mcp.tool(get_memories)
    mcp.tool(update_memory)
    mcp.tool(forget)
    mcp.tool(timeline)
    mcp.tool(reflect)
    # Knowledge graph tools
    mcp.tool(link_memories)
    mcp.tool(get_links)
    mcp.tool(traverse_graph)
    mcp.tool(suggest_links)
    # Conflict tools
    mcp.tool(get_conflicts)
    mcp.tool(resolve_conflict)
    mcp.tool(judge_conflicts)
    # Consolidation tools
    mcp.tool(consolidate)
    # Observer tools
    mcp.tool(observe)


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
