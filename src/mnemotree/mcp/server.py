from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from mnemotree.core.memory import MemoryCore, ModeDefaultsConfig, NerConfig
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


def _serialize_memory(memory: MemoryItem, *, include_embedding: bool) -> dict[str, Any]:
    data = memory.model_dump(mode="json")
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
    include_embedding: bool = False,
) -> dict[str, Any]:
    """Store a memory entry and return the stored record."""
    memory_core = await _get_memory_core()
    parsed_type = _parse_memory_type(memory_type)
    memory = await memory_core.remember(
        content=content,
        memory_type=parsed_type,
        importance=importance,
        tags=tags,
        context=context,
        analyze=analyze,
        summarize=summarize,
        references=references,
    )
    return _serialize_memory(memory, include_embedding=include_embedding)


async def recall(
    query: str,
    limit: int = 10,
    scoring: bool = True,
    update_access: bool = False,
    include_embedding: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve memories relevant to a query string."""
    memory_core = await _get_memory_core()
    memories = await memory_core.recall(
        query=query,
        limit=limit,
        scoring=scoring,
        update_access=update_access,
    )
    return [_serialize_memory(memory, include_embedding=include_embedding) for memory in memories]


async def search_index(
    query: str,
    limit: int = 10,
    include_summary: bool = True,
) -> list[dict[str, Any]]:
    """Return compact search results for progressive disclosure workflows."""
    memory_core = await _get_memory_core()
    memories = await memory_core.recall(
        query=query,
        limit=limit,
        scoring=True,
        update_access=False,
    )
    results: list[dict[str, Any]] = []
    for rank, memory in enumerate(memories, start=1):
        item = _serialize_memory_index(memory, rank)
        if not include_summary:
            item.pop("summary", None)
        results.append(item)
    return results


async def timeline(
    memory_id: str | None = None,
    timestamp: str | None = None,
    before: int = 3,
    after: int = 3,
    include_anchor: bool = True,
    include_embedding: bool = False,
) -> list[dict[str, Any]]:
    """Return memories around a given memory or timestamp in chronological order."""
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
) -> list[dict[str, Any]]:
    """Fetch full memory records by ID."""
    memory_core = await _get_memory_core()
    store = memory_core.store
    if not memory_ids:
        return []
    results = await asyncio.gather(
        *(store.get_memory(memory_id) for memory_id in memory_ids)
    )
    return [
        _serialize_memory(memory, include_embedding=include_embedding)
        for memory in results
        if memory is not None
    ]


async def forget(memory_id: str) -> bool:
    """Delete a memory by ID."""
    memory_core = await _get_memory_core()
    return await memory_core.forget(memory_id)


async def reflect(min_importance: float = 0.7) -> dict[str, Any]:
    """Summarize patterns across higher-importance memories."""
    memory_core = await _get_memory_core()
    return await memory_core.reflect(min_importance=min_importance)


def memory_types() -> list[str]:
    """Return supported memory type values."""
    return [memory_type.value for memory_type in MemoryType]


def health() -> dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}


_FASTMCP_IMPORT_ERROR = (
    "FastMCP is required to run the MCP server. Install with `mnemotree[mcp_server]`."
)
_mcp_instance: Any | None = None


def _register_tools(mcp: Any) -> None:
    mcp.tool(remember)
    mcp.tool(recall)
    mcp.tool(search_index)
    mcp.tool(timeline)
    mcp.tool(get_memories)
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
