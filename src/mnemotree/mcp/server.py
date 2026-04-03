from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemotree.core.memory import (
    AgentLayerConfig,
    MemoryCore,
    ModeDefaultsConfig,
    NerConfig,
    RecallFilters,
    RetrievalConfig,
)
from mnemotree.core.models import (
    LinkType,
    MemoryItem,
    MemoryLink,
    MemoryType,
    ObservationKind,
    ObservationStatus,
    coerce_datetime,
    compute_observation_confidence,
)
from mnemotree.ner import create_ner
from mnemotree.store.base import BaseMemoryStore
from mnemotree.store.protocols import (
    SupportsKnowledgeGraph,
    SupportsMemoryListing,
    SupportsSummaries,
)

_memory_lock = asyncio.Lock()
_memory_cores: dict[str, MemoryCore] = {}
_observation_counters: dict[str, int] = {}


async def _auto_compact(
    *,
    repo_id: str | None,
    worktree_id: str | None = None,
    task_id: str | None = None,
) -> None:
    """Fire-and-forget background compaction after observation threshold is reached."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        scope_kind = "task" if task_id else ("worktree" if worktree_id else "repo")
        await agent_compact_summary(
            repo_id=repo_id or "",
            scope_kind=scope_kind,
            worktree_id=worktree_id,
            task_id=task_id,
        )
        logger.debug("auto-compaction completed for scope=%s repo=%s", scope_kind, repo_id)
    except Exception:
        logger.debug("auto-compaction failed", exc_info=True)


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
        repo_id=filters.get("repo_id"),
        worktree_id=filters.get("worktree_id"),
        task_id=filters.get("task_id"),
        agent_id=filters.get("agent_id"),
        run_id=filters.get("run_id"),
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
        "repo_id": memory.repo_id,
        "worktree_id": memory.worktree_id,
        "task_id": memory.task_id,
        "agent_id": memory.agent_id,
        "run_id": memory.run_id,
        "observation_status": memory.observation_status.value if memory.observation_status else None,
        "observation_kind": memory.observation_kind.value if memory.observation_kind else None,
        "is_hot": memory.is_hot,
    }


def _normalize_repo_scope(
    *,
    repo_id: str | None = None,
    repo_root: str | None = None,
) -> str | None:
    if repo_id and repo_id.strip():
        return repo_id.strip()
    if repo_root and repo_root.strip():
        return os.path.abspath(os.path.expanduser(repo_root.strip()))
    return None


def _agent_store_settings(repo_scope: str | None) -> tuple[str, str]:
    if repo_scope is None:
        persist_dir = os.getenv("MNEMOTREE_MCP_PERSIST_DIR", ".mnemotree/mnemotree.sqlite")
        collection_name = os.getenv("MNEMOTREE_MCP_COLLECTION", "memories")
        return persist_dir, collection_name

    root = Path(
        os.path.expanduser(
            os.getenv("MNEMOTREE_MCP_AGENT_PERSIST_ROOT", "~/.mnemotree/agent-memory")
        )
    )
    digest = hashlib.sha1(repo_scope.encode("utf-8")).hexdigest()[:16]
    db_path = root / f"{digest}.sqlite"
    return str(db_path), "memories"


def _scope_kwargs(
    *,
    repo_id: str | None = None,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    scope = {
        "repo_id": repo_id,
        "worktree_id": worktree_id,
        "task_id": task_id,
        "agent_id": agent_id,
        "run_id": run_id,
    }
    return {key: value for key, value in scope.items() if value is not None}


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


async def _get_memory_core(
    *,
    repo_id: str | None = None,
    repo_root: str | None = None,
) -> MemoryCore:
    repo_scope = _normalize_repo_scope(repo_id=repo_id, repo_root=repo_root)
    cache_key = repo_scope or "__default__"
    async with _memory_lock:
        existing = _memory_cores.get(cache_key)
        if existing is not None:
            return existing

        persist_dir, collection_name = _agent_store_settings(repo_scope)

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

        # Profile support: "agent" profile enables agent-layer features
        profile = os.getenv("MNEMOTREE_MCP_PROFILE", "default").strip().lower()
        agent_layer_config = None
        if profile == "agent" or repo_scope is not None:
            agent_layer_config = AgentLayerConfig(
                enable_freshness_scoring=_env_bool("MNEMOTREE_MCP_ENABLE_FRESHNESS", False),
                enable_validation=_env_bool("MNEMOTREE_MCP_ENABLE_VALIDATION", False),
                graph_weight=0.15,
                temporal_weight=0.1,
            )

        memory_core = MemoryCore(
            store=store,
            mode_defaults=mode_defaults,
            ner_config=ner_config,
            retrieval_config=retrieval_config,
            default_repo_id=repo_scope,
            agent_layer_config=agent_layer_config,
        )
        _memory_cores[cache_key] = memory_core
        return memory_core


async def remember(
    content: str,
    memory_type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    repo_id: str | None = None,
    repo_root: str | None = None,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
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
    memory_core = await _get_memory_core(repo_id=repo_id, repo_root=repo_root)
    parsed_type = _parse_memory_type(memory_type)
    remember_kwargs: dict[str, Any] = {
        "content": content,
        "memory_type": parsed_type,
        "importance": importance,
        "tags": tags,
        "context": context,
        **_scope_kwargs(
            repo_id=repo_id or _normalize_repo_scope(repo_root=repo_root),
            worktree_id=worktree_id,
            task_id=task_id,
            agent_id=agent_id,
            run_id=run_id,
        ),
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
    repo_id: str | None = None,
    repo_root: str | None = None,
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
    memory_core = await _get_memory_core(repo_id=repo_id, repo_root=repo_root)
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


async def agent_remember_observation(
    repo_id: str,
    content: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    kind: str = "observation",
    observation_status: str = "tentative",
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Store a scoped agent observation in repo memory.

    Args:
        kind: One of attempt, result, decision, handoff, warning, observation.
        observation_status: One of hypothesis, tentative, confirmed, refuted.
        evidence_refs: List of evidence references (commit SHAs, file paths, test names).
    """
    merged_metadata = dict(metadata or {})
    merged_metadata["observation_status"] = observation_status
    if kind != "observation":
        try:
            merged_metadata["observation_kind"] = ObservationKind(kind).value
        except ValueError:
            merged_metadata["observation_kind"] = kind
    else:
        merged_metadata["observation_kind"] = "observation"
    if evidence_refs:
        merged_metadata["evidence_refs"] = evidence_refs
    merged_metadata.setdefault("agent_layer", True)

    # Phase B: auto-compute confidence from observation status + evidence
    try:
        obs_enum = ObservationStatus(observation_status)
    except ValueError:
        obs_enum = ObservationStatus.TENTATIVE
    computed_confidence = compute_observation_confidence(obs_enum, evidence_refs)
    merged_metadata["confidence"] = computed_confidence

    merged_tags = list(tags or [])
    if kind not in merged_tags:
        merged_tags.append(kind)

    result = await remember(
        content=content,
        memory_type=MemoryType.SEMANTIC.value,
        importance=importance,
        tags=merged_tags,
        context=context,
        metadata=merged_metadata,
        repo_id=repo_id,
        worktree_id=worktree_id,
        task_id=task_id,
        agent_id=agent_id,
        run_id=run_id,
    )

    # Phase C: auto-compaction counter
    try:
        scope_key = f"{repo_id or ''}:{worktree_id or ''}:{task_id or ''}"
        _observation_counters[scope_key] = _observation_counters.get(scope_key, 0) + 1
        memory_core = await _get_memory_core(repo_id=repo_id)
        cfg = memory_core._agent_layer_config
        if (
            getattr(cfg, "auto_compaction_enabled", False)
            and _observation_counters[scope_key] >= getattr(cfg, "auto_compaction_threshold", 10)
        ):
            _observation_counters[scope_key] = 0
            asyncio.create_task(_auto_compact(
                repo_id=repo_id,
                worktree_id=worktree_id,
                task_id=task_id,
            ))
    except Exception:
        pass  # Don't let compaction bookkeeping break observation storage

    return result


async def agent_recall_context(
    repo_id: str,
    query: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 8,
    compact: bool = True,
) -> dict[str, Any]:
    """Recall agent-scoped context for a repo, task, or worktree."""
    filters = _scope_kwargs(
        repo_id=repo_id,
        worktree_id=worktree_id,
        task_id=task_id,
        agent_id=agent_id,
        run_id=run_id,
    )
    memories = await recall(
        query=query,
        limit=limit,
        filters=filters,
        compact=compact,
        repo_id=repo_id,
    )
    return {
        "scope": filters,
        "results": memories,
    }


async def agent_recall_agentic(
    repo_id: str,
    query: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Multi-round agentic recall with gap detection and RRF fusion.

    Performs iterative retrieval: initial recall, then detects gaps in results
    and issues refined sub-queries to fill them. Returns richer results than
    single-pass agent_recall_context, especially for multi-hop queries.

    Args:
        repo_id: Repository scope identifier.
        query: Search query text.
        worktree_id: Optional worktree scope.
        task_id: Optional task scope.
        agent_id: Optional agent identity.
        run_id: Optional run identifier.
        limit: Maximum results to return (default: 8).

    Returns:
        Dict with scope, memories, rounds used, sub_queries generated, and counts.
    """
    memory_core = await _get_memory_core(repo_id=repo_id)

    from mnemotree.core.memory import RecallFilters

    scope = _scope_kwargs(
        repo_id=repo_id,
        worktree_id=worktree_id,
        task_id=task_id,
        agent_id=agent_id,
        run_id=run_id,
    )

    filters = _parse_recall_filters(scope) if scope else None

    result = await memory_core.recall_agentic(
        query=query,
        limit=limit,
        filters=filters,
    )

    # Serialize memories for MCP transport
    memories = result["memories"]
    serialized = []
    for rank, memory in enumerate(memories, start=1):
        serialized.append(_serialize_memory_index(memory, rank))

    return {
        "scope": scope,
        "results": serialized,
        "rounds": result["rounds"],
        "sub_queries": result["sub_queries"],
        "initial_count": result["initial_count"],
        "final_count": result["final_count"],
    }


async def agent_upsert_summary(
    repo_id: str,
    scope_kind: str,
    content: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
    source_memory_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update a scoped summary for agent coordination."""
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store
    if not isinstance(store, SupportsSummaries):
        raise NotImplementedError("This store does not support summaries.")
    return await store.upsert_summary(
        repo_id=repo_id,
        scope_kind=scope_kind,
        content=content,
        worktree_id=worktree_id,
        task_id=task_id,
        source_memory_ids=source_memory_ids,
        metadata=metadata,
    )


async def agent_get_summary(
    repo_id: str,
    scope_kind: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a scoped summary for repo, worktree, or task context."""
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store
    if not isinstance(store, SupportsSummaries):
        raise NotImplementedError("This store does not support summaries.")
    return await store.get_summary(
        repo_id=repo_id,
        scope_kind=scope_kind,
        worktree_id=worktree_id,
        task_id=task_id,
    )


async def agent_update_observation_status(
    repo_id: str,
    memory_id: str,
    observation_status: str,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Update the observation status of a memory (e.g. promote tentative to confirmed).

    Args:
        memory_id: The memory to update.
        observation_status: New status: hypothesis, tentative, confirmed, refuted.
        evidence_refs: Optional new evidence references to append.
    """
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store
    memory = await store.get_memory(memory_id)
    if memory is None:
        raise ValueError(f"Memory {memory_id} not found.")

    try:
        new_status = ObservationStatus(observation_status)
    except ValueError:
        raise ValueError(
            f"Invalid observation_status '{observation_status}'. "
            f"Must be one of: {', '.join(s.value for s in ObservationStatus)}"
        )

    updates: dict[str, Any] = {"observation_status": new_status.value}
    all_evidence = list(memory.evidence_refs or [])
    if evidence_refs:
        all_evidence.extend(evidence_refs)
        updates["evidence_refs"] = all_evidence

    # Phase B: auto-compute confidence from new status + all evidence
    updates["confidence"] = compute_observation_confidence(new_status, all_evidence)

    from mnemotree.store.protocols import SupportsMetadataUpdate

    if not isinstance(store, SupportsMetadataUpdate):
        raise NotImplementedError("Store does not support metadata updates.")
    await store.update_memory_metadata(memory_id, updates)
    return {"memory_id": memory_id, "observation_status": new_status.value, "updated": True}


async def agent_recall_with_summary(
    repo_id: str,
    query: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 8,
    scope_kind: str = "task",
    exclude_refuted: bool = True,
) -> dict[str, Any]:
    """Summary-first recall: returns summary + hot observations + top-k matches.

    This is the primary agent recall endpoint for fresh sessions:
    1. Load the relevant summary for the scope
    2. Fetch pinned HOT observations
    3. Retrieve top-k matching observations
    4. Exclude refuted and coordination records
    """
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store

    # Step 1: Load summary
    summary_data = None
    if isinstance(store, SupportsSummaries):
        summary_data = await store.get_summary(
            repo_id=repo_id,
            scope_kind=scope_kind,
            worktree_id=worktree_id,
            task_id=task_id,
        )

    # Step 2: Recall scoped observations
    from mnemotree.core.memory import RecallFilters

    filters_dict = _scope_kwargs(
        repo_id=repo_id,
        worktree_id=worktree_id,
        task_id=task_id,
        agent_id=agent_id,
        run_id=run_id,
    )
    memories = await recall(
        query=query,
        limit=limit * 2,  # fetch extra for filtering
        filters=filters_dict,
        compact=False,
        repo_id=repo_id,
    )

    # Step 3: Separate hot observations and filter
    hot_observations = []
    regular_observations = []
    for mem in memories:
        # Exclude refuted
        obs_status = mem.get("observation_status") or (mem.get("metadata") or {}).get(
            "observation_status"
        )
        if exclude_refuted and obs_status == "refuted":
            continue
        if mem.get("is_hot"):
            hot_observations.append(mem)
        else:
            regular_observations.append(mem)

    # Trim regular to limit
    regular_observations = regular_observations[:limit]

    # Optional: apply freshness scoring if enabled
    if memory_core._agent_layer_config.enable_freshness_scoring:
        # Re-retrieve as MemoryItems for freshness scoring
        # (observations are already dicts from recall, skip for now)
        pass

    result: dict[str, Any] = {
        "scope": filters_dict,
        "summary": summary_data,
        "hot_observations": hot_observations,
        "observations": regular_observations,
        "total_results": len(hot_observations) + len(regular_observations),
    }

    # Phase 6 validation: annotate observations with evidence validation status
    if memory_core._agent_layer_config.enable_validation:
        for obs in hot_observations + regular_observations:
            refs = obs.get("evidence_refs") or (obs.get("metadata") or {}).get("evidence_refs")
            if refs and isinstance(refs, list):
                from mnemotree.core.models import MemoryItem, MemoryType

                _temp = MemoryItem(
                    content="", memory_type=MemoryType.SEMANTIC, importance=0.5, evidence_refs=refs,
                )
                obs["evidence_validation"] = memory_core.validate_evidence_refs(_temp)

    return result


async def agent_compact_summary(
    repo_id: str,
    scope_kind: str,
    query: str = "recent activity",
    worktree_id: str | None = None,
    task_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Compact recent observations into a summary for the given scope.

    Fetches recent observations, synthesizes a summary, and stores it.
    This is the compaction path for summary-first retrieval.
    """
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store

    if not isinstance(store, SupportsSummaries):
        raise NotImplementedError("This store does not support summaries.")

    # Fetch recent observations for this scope
    filters_dict = _scope_kwargs(
        repo_id=repo_id,
        worktree_id=worktree_id,
        task_id=task_id,
    )
    memories = await recall(
        query=query,
        limit=limit,
        filters=filters_dict,
        compact=False,
        repo_id=repo_id,
    )

    if not memories:
        return {"compacted": False, "reason": "no observations to compact"}

    # Synthesize summary from observations
    memory_ids = []
    content_parts = []
    for mem in memories:
        mid = mem.get("memory_id", "")
        memory_ids.append(mid)
        snippet = mem.get("summary") or mem.get("content", "")
        obs_status = mem.get("observation_status") or (mem.get("metadata") or {}).get(
            "observation_status", ""
        )
        kind = mem.get("observation_kind") or (mem.get("metadata") or {}).get("kind", "")
        prefix = f"[{kind}]" if kind else ""
        suffix = f"({obs_status})" if obs_status else ""
        content_parts.append(f"{prefix} {snippet} {suffix}".strip())

    summary_content = "\n".join(f"- {part}" for part in content_parts)

    result = await store.upsert_summary(
        repo_id=repo_id,
        scope_kind=scope_kind,
        content=summary_content,
        worktree_id=worktree_id,
        task_id=task_id,
        source_memory_ids=memory_ids,
    )

    return {"compacted": True, "summary": result, "observation_count": len(memories)}


async def agent_inspect_memories(
    repo_id: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    observation_status: str | None = None,
    observation_kind: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Audit/inspect agent memories for a given scope (Phase 8: human surface).

    Returns a structured list of memories with their observation metadata.
    """
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store

    if not isinstance(store, SupportsMemoryListing):
        raise NotImplementedError("This store does not support memory listing.")

    all_memories = await store.list_memories(include_embeddings=False)

    # Filter by scope
    results = []
    for mem in all_memories:
        if repo_id and mem.repo_id != repo_id:
            continue
        if worktree_id and mem.worktree_id != worktree_id:
            continue
        if task_id and mem.task_id != task_id:
            continue
        if agent_id and mem.agent_id != agent_id:
            continue
        if observation_status:
            mem_status = mem.observation_status.value if mem.observation_status else None
            if mem_status != observation_status:
                continue
        if observation_kind:
            mem_kind = mem.observation_kind.value if mem.observation_kind else None
            if mem_kind != observation_kind:
                continue

        results.append({
            "memory_id": mem.memory_id,
            "content": mem.content[:200],
            "summary": mem.summary,
            "memory_type": mem.memory_type.value,
            "importance": mem.importance,
            "observation_status": mem.observation_status.value if mem.observation_status else None,
            "observation_kind": mem.observation_kind.value if mem.observation_kind else None,
            "evidence_refs": mem.evidence_refs,
            "is_hot": mem.is_hot,
            "tags": mem.tags,
            "timestamp": mem.model_dump(mode="json").get("timestamp"),
            "agent_id": mem.agent_id,
            "worktree_id": mem.worktree_id,
            "task_id": mem.task_id,
        })
        if len(results) >= limit:
            break

    return {"count": len(results), "memories": results}


async def agent_inspect_summaries(
    repo_id: str,
    worktree_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Inspect all summaries for a given scope (Phase 8: human surface)."""
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store

    if not isinstance(store, SupportsSummaries):
        raise NotImplementedError("This store does not support summaries.")

    # Check common scope kinds
    scope_kinds = ["repo", "worktree", "task", "sprint", "feature"]
    summaries = []
    for sk in scope_kinds:
        result = await store.get_summary(
            repo_id=repo_id,
            scope_kind=sk,
            worktree_id=worktree_id,
            task_id=task_id,
        )
        if result:
            summaries.append(result)

    return {"count": len(summaries), "summaries": summaries}


async def agent_delete_memory(
    repo_id: str,
    memory_id: str,
) -> dict[str, Any]:
    """Delete a specific memory by ID (Phase 8: human surface)."""
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store
    deleted = await store.delete_memory(memory_id)
    return {"memory_id": memory_id, "deleted": deleted}


async def agent_correct_memory(
    repo_id: str,
    memory_id: str,
    content: str | None = None,
    observation_status: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Correct or update an existing memory (Phase 8: human surface).

    Allows humans to fix observation content, status, importance, or evidence.
    """
    memory_core = await _get_memory_core(repo_id=repo_id)
    store = memory_core.store
    memory = await store.get_memory(memory_id)
    if memory is None:
        raise ValueError(f"Memory {memory_id} not found.")

    from mnemotree.store.protocols import SupportsMetadataUpdate

    if not isinstance(store, SupportsMetadataUpdate):
        raise NotImplementedError("Store does not support updates.")

    updates: dict[str, Any] = {}
    if content is not None:
        updates["content"] = content
    if observation_status is not None:
        updates["observation_status"] = ObservationStatus(observation_status).value
    if importance is not None:
        updates["importance"] = importance
    if tags is not None:
        updates["tags"] = tags
    if evidence_refs is not None:
        updates["evidence_refs"] = evidence_refs

    if updates:
        await store.update_memory_metadata(memory_id, updates)

    return {"memory_id": memory_id, "updated_fields": list(updates.keys())}


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
    parsed_types = (
        [_parse_link_type(lt) for lt in link_types]
        if link_types
        else None
    )
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
                "target_id": target_memory.memory_id if isinstance(target_memory, MemoryItem) else str(target_memory),
                "score": float(score),
            }
            if link_type is not None:
                entry["link_type"] = link_type.value if hasattr(link_type, "value") else str(link_type)
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
                result["conflicting_memories"].append({
                    "memory_id": cm.memory_id,
                    "snippet": _memory_snippet(cm),
                    "importance": cm.importance,
                    "memory_type": cm.memory_type.value,
                })

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
        results = await asyncio.gather(
            *(memory_core.store.get_memory(mid) for mid in memory_ids)
        )
        memories = [m for m in results if m is not None]
    elif query:
        memories = await memory_core.recall(query, limit=limit)
    else:
        raise ValueError("Provide either memory_ids or query.")

    conflicts = memory_core.judge_conflicts(
        memories, similarity_threshold=similarity_threshold
    )
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
    mcp.tool(agent_remember_observation)
    mcp.tool(agent_recall_context)
    mcp.tool(agent_upsert_summary)
    mcp.tool(agent_get_summary)
    mcp.tool(agent_update_observation_status)
    mcp.tool(agent_recall_agentic)
    mcp.tool(agent_recall_with_summary)
    mcp.tool(agent_compact_summary)
    # Human surface tools (Phase 8)
    mcp.tool(agent_inspect_memories)
    mcp.tool(agent_inspect_summaries)
    mcp.tool(agent_delete_memory)
    mcp.tool(agent_correct_memory)
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
