"""Tests for MCP server helpers."""

from __future__ import annotations

import builtins
import re
import runpy
import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from mnemotree.core.memory import ModeDefaultsConfig, NerConfig, RetrievalConfig
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.mcp import server


def _make_memory(
    memory_id: str,
    timestamp: datetime,
    *,
    summary: str | None = None,
    embedding: list[float] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=f"content-{memory_id}",
        summary=summary,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp=timestamp,
        last_accessed=timestamp,
        embedding=embedding,
        tags=["tag"],
    )


def test_env_bool_default_and_values(monkeypatch):
    monkeypatch.delenv("MNEMO_BOOL", raising=False)
    assert server._env_bool("MNEMO_BOOL", True) is True

    monkeypatch.setenv("MNEMO_BOOL", "0")
    assert server._env_bool("MNEMO_BOOL", True) is False

    monkeypatch.setenv("MNEMO_BOOL", "YES")
    assert server._env_bool("MNEMO_BOOL", False) is True


def test_ensure_list_none():
    assert server._ensure_list(None) is None


def test_ensure_list_already_list():
    assert server._ensure_list([1, 2, 3]) == [1, 2, 3]


def test_ensure_list_scalar_to_list():
    assert server._ensure_list("single") == ["single"]
    assert server._ensure_list(42) == [42]


def test_coerce_tags_none():
    assert server._coerce_tags(None) == []


def test_coerce_tags_list():
    assert server._coerce_tags(["a", "b"]) == ["a", "b"]


def test_coerce_tags_single_value():
    assert server._coerce_tags("single") == ["single"]


def test_coerce_tags_converts_to_string():
    assert server._coerce_tags([1, 2, 3]) == ["1", "2", "3"]


def test_parse_memory_type():
    assert server._parse_memory_type(None) is None
    assert server._parse_memory_type(" semantic ") == MemoryType.SEMANTIC
    assert server._parse_memory_type("EPISODIC") == MemoryType.EPISODIC
    with pytest.raises(ValueError, match="Unknown memory_type"):
        server._parse_memory_type("unknown")


def test_parse_recall_filters_empty():
    assert server._parse_recall_filters(None) is None
    assert server._parse_recall_filters({}) is None


def test_parse_recall_filters_memory_types_list():
    filters = server._parse_recall_filters({"memory_types": ["semantic", "episodic"]})
    assert filters is not None
    assert filters.memory_types == [MemoryType.SEMANTIC, MemoryType.EPISODIC]


def test_parse_recall_filters_memory_type_single():
    filters = server._parse_recall_filters({"memory_type": "semantic"})
    assert filters is not None
    assert filters.memory_types == [MemoryType.SEMANTIC]


def test_parse_recall_filters_memory_type_enum_passthrough():
    filters = server._parse_recall_filters({"memory_types": [MemoryType.PROCEDURAL]})
    assert filters is not None
    assert filters.memory_types == [MemoryType.PROCEDURAL]


def test_parse_recall_filters_tags_single():
    filters = server._parse_recall_filters({"tags": "single-tag"})
    assert filters is not None
    assert filters.tags == ["single-tag"]


def test_parse_recall_filters_tags_list():
    filters = server._parse_recall_filters({"tags": ["a", "b"]})
    assert filters is not None
    assert filters.tags == ["a", "b"]


def test_parse_recall_filters_all_fields():
    filters = server._parse_recall_filters(
        {
            "memory_types": ["semantic"],
            "tags": ["tag1"],
            "min_importance": 0.5,
            "max_importance": 0.9,
            "since": "2024-01-01",
            "until": "2024-12-31",
            "source": "test",
            "author": "user",
            "conversation_id": "conv-1",
            "user_id": "user-1",
        }
    )
    assert filters is not None
    assert filters.memory_types == [MemoryType.SEMANTIC]
    assert filters.tags == ["tag1"]
    assert filters.min_importance == 0.5
    assert filters.max_importance == 0.9
    assert filters.since == "2024-01-01"
    assert filters.until == "2024-12-31"
    assert filters.source == "test"
    assert filters.author == "user"
    assert filters.conversation_id == "conv-1"
    assert filters.user_id == "user-1"


def test_serialize_memory_excludes_embedding():
    memory = _make_memory(
        "mem-1",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        embedding=[0.1, 0.2],
    )
    data = server._serialize_memory(memory, include_embedding=False)
    assert data["memory_id"] == "mem-1"
    assert "embedding" not in data


def test_serialize_memory_with_fields_filter():
    memory = _make_memory(
        "mem-fields",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        summary="test summary",
        embedding=[0.1, 0.2],
    )
    data = server._serialize_memory(
        memory,
        include_embedding=True,
        fields=["memory_id", "content", "nonexistent"],
    )
    assert data == {"memory_id": "mem-fields", "content": "content-mem-fields"}
    assert "summary" not in data
    assert "embedding" not in data


def test_memory_snippet_uses_summary_and_truncates():
    memory = _make_memory(
        "mem-2",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        summary="abcdefghij",
    )
    snippet = server._memory_snippet(memory, max_len=8)
    assert snippet == "abcde..."


def test_serialize_memory_index_fields():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-3", ts, summary="summary")
    data = server._serialize_memory_index(memory, rank=2)
    assert data["memory_id"] == "mem-3"
    assert data["rank"] == 2
    assert data["summary"] == "summary"
    assert data["snippet"] == "summary"
    assert data["memory_type"] == "semantic"
    assert abs(data["importance"] - 0.5) < 1e-9
    assert data["tags"] == ["tag"]
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_timeline_requires_anchor():
    with pytest.raises(ValueError, match="either memory_id or timestamp"):
        await server.timeline()


@pytest.mark.asyncio
async def test_timeline_missing_memory_id_returns_empty(monkeypatch):
    memories = [
        _make_memory("mem-1", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _make_memory("mem-2", datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ]

    async def _fake_get_all_memories(_, *, include_embeddings):
        return memories

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    result = await server.timeline(memory_id="missing")
    assert result == []


@pytest.mark.asyncio
async def test_timeline_invalid_timestamp_raises(monkeypatch):
    memories = [_make_memory("mem-1", datetime(2024, 1, 1, tzinfo=timezone.utc))]

    async def _fake_get_all_memories(_, *, include_embeddings):
        return memories

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    with pytest.raises(ValueError, match="Invalid timestamp format"):
        await server.timeline(timestamp="not-a-date")


@pytest.mark.asyncio
async def test_timeline_offsets_and_embeddings(monkeypatch):
    memories = [
        _make_memory("mem-1", datetime(2024, 1, 1, tzinfo=timezone.utc), embedding=[1.0]),
        _make_memory("mem-2", datetime(2024, 1, 2, tzinfo=timezone.utc), embedding=[2.0]),
        _make_memory("mem-3", datetime(2024, 1, 3, tzinfo=timezone.utc), embedding=[3.0]),
    ]

    async def _fake_get_all_memories(_, *, include_embeddings):
        return memories

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    results = await server.timeline(
        memory_id="mem-2",
        before=1,
        after=1,
        include_anchor=True,
        include_embedding=True,
    )

    assert [item["memory_id"] for item in results] == ["mem-1", "mem-2", "mem-3"]
    assert [item["offset"] for item in results] == [-1, 0, 1]
    assert results[1]["anchor"] is True
    assert results[0]["embedding"] == [1.0]
    assert results[1]["embedding"] == [2.0]
    assert results[2]["embedding"] == [3.0]


@pytest.mark.asyncio
async def test_timeline_excludes_anchor(monkeypatch):
    memories = [
        _make_memory("mem-1", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _make_memory("mem-2", datetime(2024, 1, 2, tzinfo=timezone.utc)),
        _make_memory("mem-3", datetime(2024, 1, 3, tzinfo=timezone.utc)),
    ]

    async def _fake_get_all_memories(_, *, include_embeddings):
        return memories

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    results = await server.timeline(
        memory_id="mem-2",
        before=1,
        after=1,
        include_anchor=False,
    )

    assert [item["memory_id"] for item in results] == ["mem-1", "mem-3"]
    assert all("anchor" not in item for item in results)


def test_load_fastmcp_import_error(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "fastmcp", raising=False)

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastmcp":
            raise ModuleNotFoundError("No module named 'fastmcp'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(ModuleNotFoundError, match=re.escape(server._FASTMCP_IMPORT_ERROR)):
        server._load_fastmcp()


def test_get_mcp_registers_tools(monkeypatch):
    class DummyFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: list[object] = []

        def tool(self, func):
            self.tools.append(func)
            return func

        def run(self) -> None:
            """Mock implementation - no-op for testing."""

    fastmcp_module = types.ModuleType("fastmcp")
    fastmcp_module.FastMCP = DummyFastMCP

    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_module)
    monkeypatch.setattr(server, "_mcp_instance", None)

    instance = server._get_mcp()

    assert isinstance(instance, DummyFastMCP)
    assert len(instance.tools) == 5


def test_memory_timestamp_fallbacks():
    dummy = types.SimpleNamespace(timestamp=object(), last_accessed=None)
    result = server._memory_timestamp(dummy)
    assert result == datetime.min.replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_get_all_memories_requires_list_memories():
    class DummyCore:
        def __init__(self) -> None:
            self.store = object()

    with pytest.raises(NotImplementedError, match="list_memories"):
        await server._get_all_memories(DummyCore(), include_embeddings=False)


@pytest.mark.asyncio
async def test_get_all_memories_lists_memories():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-list", ts)

    class DummyStore:
        async def list_memories(self, *, include_embeddings: bool = False):
            assert include_embeddings is True
            return [memory]

    class DummyCore:
        def __init__(self) -> None:
            self.store = DummyStore()

    result = await server._get_all_memories(DummyCore(), include_embeddings=True)
    assert result == [memory]


@pytest.mark.asyncio
async def test_get_memory_core_remote_store_and_ner(monkeypatch):
    class DummyStore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

    class DummyMemoryCore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.store = kwargs["store"]

    fake_store_module = types.ModuleType("mnemotree.store")
    fake_store_module.ChromaMemoryStore = DummyStore

    ner_calls: list[tuple[str, dict[str, str]]] = []

    def _fake_create_ner(backend: str, **kwargs) -> str:
        ner_calls.append((backend, kwargs))
        return "ner-instance"

    monkeypatch.setitem(sys.modules, "mnemotree.store", fake_store_module)
    monkeypatch.setattr(server, "MemoryCore", DummyMemoryCore)
    monkeypatch.setattr(server, "create_ner", _fake_create_ner)
    monkeypatch.setattr(server, "_memory_core", None)

    monkeypatch.setenv("MNEMOTREE_MCP_CHROMA_HOST", "localhost")
    monkeypatch.setenv("MNEMOTREE_MCP_CHROMA_PORT", "1234")
    monkeypatch.setenv("MNEMOTREE_MCP_CHROMA_SSL", "yes")
    monkeypatch.setenv("MNEMOTREE_MCP_COLLECTION", "memories-test")
    monkeypatch.setenv("MNEMOTREE_MCP_NER_BACKEND", "gliner")
    monkeypatch.setenv("MNEMOTREE_MCP_NER_MODEL", "ner-model")
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_NER", "1")
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_KEYWORDS", "true")

    core = await server._get_memory_core()

    assert isinstance(core, DummyMemoryCore)
    assert core.store.kwargs == {
        "host": "localhost",
        "port": 1234,
        "ssl": True,
        "collection_name": "memories-test",
    }
    assert core.store.initialized is True
    assert ner_calls == [("gliner", {"model_name": "ner-model"})]
    mode_defaults = core.kwargs["mode_defaults"]
    ner_config = core.kwargs["ner_config"]
    assert isinstance(mode_defaults, ModeDefaultsConfig)
    assert mode_defaults.mode == "lite"
    assert mode_defaults.enable_keywords is True
    assert isinstance(ner_config, NerConfig)
    assert ner_config.ner == "ner-instance"
    assert ner_config.enable_ner is True


@pytest.mark.asyncio
async def test_get_memory_core_persist_dir(monkeypatch):
    class DummyStore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def initialize(self) -> None:
            return None

    class DummyMemoryCore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.store = kwargs["store"]

    fake_store_module = types.ModuleType("mnemotree.store")
    fake_store_module.ChromaMemoryStore = DummyStore

    def _fake_create_ner(*args, **kwargs) -> None:
        raise AssertionError("create_ner should not be called without backend config")

    monkeypatch.setitem(sys.modules, "mnemotree.store", fake_store_module)
    monkeypatch.setattr(server, "MemoryCore", DummyMemoryCore)
    monkeypatch.setattr(server, "create_ner", _fake_create_ner)
    monkeypatch.setattr(server, "_memory_core", None)

    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_HOST", raising=False)
    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_PORT", raising=False)
    monkeypatch.setenv("MNEMOTREE_MCP_PERSIST_DIR", "/tmp/mnemotree")
    monkeypatch.setenv("MNEMOTREE_MCP_COLLECTION", "memories-local")

    core = await server._get_memory_core()

    assert core.store.kwargs == {
        "persist_directory": "/tmp/mnemotree",
        "collection_name": "memories-local",
    }
    mode_defaults = core.kwargs["mode_defaults"]
    ner_config = core.kwargs["ner_config"]
    assert isinstance(mode_defaults, ModeDefaultsConfig)
    assert mode_defaults.enable_keywords is False
    assert isinstance(ner_config, NerConfig)
    assert ner_config.ner is None
    assert ner_config.enable_ner is False


@pytest.mark.asyncio
async def test_get_memory_core_retrieval_config_bm25(monkeypatch):
    class DummyStore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def initialize(self) -> None:
            return None

    class DummyMemoryCore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.store = kwargs["store"]

    fake_store_module = types.ModuleType("mnemotree.store")
    fake_store_module.ChromaMemoryStore = DummyStore

    def _fake_create_ner(*args, **kwargs) -> None:
        raise AssertionError("create_ner should not be called")

    monkeypatch.setitem(sys.modules, "mnemotree.store", fake_store_module)
    monkeypatch.setattr(server, "MemoryCore", DummyMemoryCore)
    monkeypatch.setattr(server, "create_ner", _fake_create_ner)
    monkeypatch.setattr(server, "_memory_core", None)

    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_HOST", raising=False)
    monkeypatch.setenv("MNEMOTREE_MCP_PERSIST_DIR", "/tmp/test")
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_BM25", "1")

    core = await server._get_memory_core()

    retrieval_config = core.kwargs["retrieval_config"]
    assert isinstance(retrieval_config, RetrievalConfig)
    assert retrieval_config.retrieval_mode == "hybrid"
    assert retrieval_config.enable_bm25 is True


@pytest.mark.asyncio
async def test_get_memory_core_retrieval_config_bm25_disabled(monkeypatch):
    class DummyStore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def initialize(self) -> None:
            return None

    class DummyMemoryCore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.store = kwargs["store"]

    fake_store_module = types.ModuleType("mnemotree.store")
    fake_store_module.ChromaMemoryStore = DummyStore

    def _fake_create_ner(*args, **kwargs) -> None:
        raise AssertionError("create_ner should not be called")

    monkeypatch.setitem(sys.modules, "mnemotree.store", fake_store_module)
    monkeypatch.setattr(server, "MemoryCore", DummyMemoryCore)
    monkeypatch.setattr(server, "create_ner", _fake_create_ner)
    monkeypatch.setattr(server, "_memory_core", None)

    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_HOST", raising=False)
    monkeypatch.setenv("MNEMOTREE_MCP_PERSIST_DIR", "/tmp/test")
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_BM25", "0")

    core = await server._get_memory_core()

    retrieval_config = core.kwargs["retrieval_config"]
    assert isinstance(retrieval_config, RetrievalConfig)
    assert retrieval_config.enable_bm25 is False


@pytest.mark.asyncio
async def test_get_memory_core_ner_model_non_gliner(monkeypatch):
    class DummyStore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def initialize(self) -> None:
            return None

    class DummyMemoryCore:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.store = kwargs["store"]

    fake_store_module = types.ModuleType("mnemotree.store")
    fake_store_module.ChromaMemoryStore = DummyStore
    ner_calls: list[tuple[str, dict[str, str]]] = []

    def _fake_create_ner(backend: str, **kwargs) -> str:
        ner_calls.append((backend, kwargs))
        return "ner-instance"

    monkeypatch.setitem(sys.modules, "mnemotree.store", fake_store_module)
    monkeypatch.setattr(server, "MemoryCore", DummyMemoryCore)
    monkeypatch.setattr(server, "create_ner", _fake_create_ner)
    monkeypatch.setattr(server, "_memory_core", None)

    monkeypatch.setenv("MNEMOTREE_MCP_NER_BACKEND", "spacy")
    monkeypatch.setenv("MNEMOTREE_MCP_NER_MODEL", "ner-model")

    core = await server._get_memory_core()

    assert isinstance(core, DummyMemoryCore)
    assert ner_calls == [("spacy", {"model": "ner-model"})]


@pytest.mark.asyncio
async def test_get_memory_core_cached_instance(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(server, "_memory_core", sentinel)

    core = await server._get_memory_core()

    assert core is sentinel


@pytest.mark.asyncio
async def test_get_memory_core_import_error(monkeypatch):
    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mnemotree.store":
            raise ModuleNotFoundError("No module named 'mnemotree.store'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setattr(server, "_memory_core", None)

    with pytest.raises(ModuleNotFoundError, match="ChromaMemoryStore is required"):
        await server._get_memory_core()


@pytest.mark.asyncio
async def test_remember_recall_compact(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-remember", ts, summary="summary", embedding=[0.3])

    memory_core = MagicMock()
    memory_core.remember = AsyncMock(return_value=memory)
    memory_core.recall = AsyncMock(return_value=[memory])

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    stored = await server.remember(
        content="hello",
        memory_type="episodic",
        importance=0.9,
        tags=["tag"],
        context={"source": "test"},
    )

    assert stored["memory_id"] == "mem-remember"
    assert "embedding" not in stored
    memory_core.remember.assert_awaited_once_with(
        content="hello",
        memory_type=MemoryType.EPISODIC,
        importance=0.9,
        tags=["tag"],
        context={"source": "test"},
    )

    recalled = await server.recall(
        query="hello",
        limit=5,
        compact=False,
    )
    assert "embedding" not in recalled[0]

    results = await server.recall(query="hello", limit=3, compact=True, include_summary=False)
    assert "summary" not in results[0]
    assert results[0]["rank"] == 1

    memory_core.recall.assert_has_awaits(
        [
            call(query="hello", limit=5, scoring=True, update_access=False),
            call(query="hello", limit=3, scoring=True, update_access=False),
        ]
    )
    assert memory_core.recall.await_count == 2


@pytest.mark.asyncio
async def test_remember_with_metadata(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-meta", ts)
    memory.metadata = {"key": "value"}

    memory_core = MagicMock()
    memory_core.remember = AsyncMock(return_value=memory)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    stored = await server.remember(
        content="test content",
        metadata={"key": "value"},
    )

    assert stored["metadata"]["key"] == "value"
    memory_core.remember.assert_awaited_once()
    call_kwargs = memory_core.remember.call_args.kwargs
    assert call_kwargs["metadata"] == {"key": "value"}


@pytest.mark.asyncio
async def test_recall_with_filters(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-filtered", ts)

    memory_core = MagicMock()
    memory_core.recall = AsyncMock(return_value=[memory])

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    results = await server.recall(
        query="test",
        limit=5,
        filters={"tags": ["important"], "min_importance": 0.5},
    )

    assert len(results) == 1
    memory_core.recall.assert_awaited_once()
    call_kwargs = memory_core.recall.call_args.kwargs
    assert "filters" in call_kwargs
    assert call_kwargs["filters"].tags == ["important"]
    assert call_kwargs["filters"].min_importance == 0.5


@pytest.mark.asyncio
async def test_update_memory_applies_patch_and_reembed(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-remember", ts, summary="summary", embedding=[0.3])
    memory.metadata = {"old": 1}

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.store_memory = AsyncMock()

    memory_core = MagicMock()
    memory_core.store = store
    memory_core.get_embedding = AsyncMock(return_value=[0.9])

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    updated = await server.update_memory(
        memory_id="mem-remember",
        patch={"content": "updated", "tags": ["new"], "metadata": {"k": "v"}},
        reembed=True,
    )

    assert updated["content"] == "updated"
    assert updated["tags"] == ["new"]
    assert updated["metadata"]["old"] == 1
    assert updated["metadata"]["k"] == "v"
    assert "embedding" not in updated

    memory_core.get_embedding.assert_awaited_once_with("updated")
    store.store_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_memory_empty_patch_raises():
    with pytest.raises(ValueError, match="patch is required"):
        await server.update_memory(memory_id="mem-1", patch={})


@pytest.mark.asyncio
async def test_update_memory_unknown_fields_raises(monkeypatch):
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    with pytest.raises(ValueError, match="Unknown update fields.*invalid_field"):
        await server.update_memory(memory_id="mem-1", patch={"invalid_field": "value"})


@pytest.mark.asyncio
async def test_update_memory_not_found_raises(monkeypatch):
    store = MagicMock()
    store.get_memory = AsyncMock(return_value=None)
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="Memory not found"):
        await server.update_memory(memory_id="missing", patch={"content": "new"})


@pytest.mark.asyncio
async def test_update_memory_importance_null_raises(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts)

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="importance cannot be null"):
        await server.update_memory(memory_id="mem-1", patch={"importance": None})


@pytest.mark.asyncio
async def test_update_memory_importance_out_of_range_raises(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts)

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="importance must be between 0 and 1"):
        await server.update_memory(memory_id="mem-1", patch={"importance": 1.5})


@pytest.mark.asyncio
async def test_update_memory_metadata_invalid_type_raises(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts)
    memory.metadata = {}

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    with pytest.raises(ValueError, match="metadata must be a dict"):
        await server.update_memory(memory_id="mem-1", patch={"metadata": "not-a-dict"})


@pytest.mark.asyncio
async def test_update_memory_metadata_null_clears(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts)
    memory.metadata = {"old": "value"}

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.store_memory = AsyncMock()
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    updated = await server.update_memory(memory_id="mem-1", patch={"metadata": None})

    assert updated["metadata"] == {}


@pytest.mark.asyncio
async def test_update_memory_summary_and_context(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts, summary="old summary")
    memory.context = {"old": "context"}
    memory.metadata = {}

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.store_memory = AsyncMock()
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    updated = await server.update_memory(
        memory_id="mem-1",
        patch={"summary": "new summary", "context": {"new": "context"}},
    )

    assert updated["summary"] == "new summary"
    assert updated["context"] == {"new": "context"}


@pytest.mark.asyncio
async def test_update_memory_no_reembed(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts, embedding=[0.1, 0.2])
    memory.metadata = {}

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.store_memory = AsyncMock()
    memory_core = MagicMock(store=store)
    memory_core.get_embedding = AsyncMock()

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    await server.update_memory(
        memory_id="mem-1",
        patch={"content": "updated content"},
        reembed=False,
    )

    memory_core.get_embedding.assert_not_awaited()
    store.store_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_memory_valid_importance(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-1", ts)
    memory.metadata = {}

    store = MagicMock()
    store.get_memory = AsyncMock(return_value=memory)
    store.store_memory = AsyncMock()
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    updated = await server.update_memory(
        memory_id="mem-1",
        patch={"importance": 0.75},
    )

    assert updated["importance"] == 0.75
    store.store_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeline_empty_memories(monkeypatch):
    async def _fake_get_all_memories(_, *, include_embeddings):
        return []

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    assert await server.timeline(memory_id="mem-1") == []


@pytest.mark.asyncio
async def test_timeline_empty_memories_with_timestamp(monkeypatch):
    async def _fake_get_all_memories(_, *, include_embeddings):
        return []

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    assert await server.timeline(timestamp="2024-01-01T00:00:00+00:00") == []


@pytest.mark.asyncio
async def test_timeline_timestamp_after_all(monkeypatch):
    memories = [
        _make_memory("mem-1", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _make_memory("mem-2", datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ]

    async def _fake_get_all_memories(_, *, include_embeddings):
        return memories

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    results = await server.timeline(timestamp="2024-02-01T00:00:00+00:00", before=0, after=0)

    assert [item["memory_id"] for item in results] == ["mem-2"]
    assert results[0]["anchor"] is True
    assert results[0]["offset"] == 0


@pytest.mark.asyncio
async def test_timeline_timestamp_between(monkeypatch):
    memories = [
        _make_memory("mem-1", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _make_memory("mem-2", datetime(2024, 1, 2, tzinfo=timezone.utc)),
        _make_memory("mem-3", datetime(2024, 1, 3, tzinfo=timezone.utc)),
    ]

    async def _fake_get_all_memories(_, *, include_embeddings):
        return memories

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    results = await server.timeline(timestamp="2024-01-02T12:00:00+00:00", before=0, after=0)

    assert [item["memory_id"] for item in results] == ["mem-3"]
    assert results[0]["anchor"] is True


@pytest.mark.asyncio
async def test_get_memories_filters_missing(monkeypatch):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    memory = _make_memory("mem-keep", ts)

    store = MagicMock()
    store.get_memory = AsyncMock(side_effect=[memory, None])
    memory_core = MagicMock(store=store)

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    results = await server.get_memories(["mem-keep", "mem-missing"])

    assert [item["memory_id"] for item in results] == ["mem-keep"]
    store.get_memory.assert_has_awaits([call("mem-keep"), call("mem-missing")])


@pytest.mark.asyncio
async def test_get_memories_empty_list(monkeypatch):
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    assert await server.get_memories([]) == []


@pytest.mark.asyncio
async def test_forget_reflect_and_helpers(monkeypatch):
    memory_core = MagicMock()
    memory_core.forget = AsyncMock(return_value=True)
    memory_core.reflect = AsyncMock(return_value={"summary": "ok"})

    monkeypatch.setattr(server, "_get_memory_core", AsyncMock(return_value=memory_core))

    assert await server.forget("mem-id") is True
    assert await server.reflect(min_importance=0.95) == {"summary": "ok"}
    assert server.memory_types() == [item.value for item in MemoryType]
    assert server.health() == {"status": "ok"}


def test_get_mcp_returns_cached_instance(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(server, "_mcp_instance", sentinel)
    monkeypatch.setattr(
        server,
        "_load_fastmcp",
        lambda: (_ for _ in ()).throw(AssertionError("should not reload")),
    )

    assert server._get_mcp() is sentinel


def test_lazy_mcp_proxies_attributes(monkeypatch):
    class DummyMCP:
        def __init__(self) -> None:
            self.value = "ok"

    dummy = DummyMCP()
    monkeypatch.setattr(server, "_get_mcp", lambda: dummy)

    proxy = server._LazyMCP()
    assert proxy.value == "ok"
    assert "value" in dir(proxy)


def test_main_runs_fastmcp(monkeypatch):
    class DummyMCP:
        def __init__(self) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

    dummy = DummyMCP()
    monkeypatch.setattr(server, "_get_mcp", lambda: dummy)

    server.main()
    assert dummy.ran is True


def test_module_main_executes(monkeypatch):
    class DummyFastMCP:
        last_instance = None

        def __init__(self, name: str) -> None:
            self.name = name
            self.ran = False
            DummyFastMCP.last_instance = self

        def tool(self, func):
            return func

        def run(self) -> None:
            self.ran = True

    fastmcp_module = types.ModuleType("fastmcp")
    fastmcp_module.FastMCP = DummyFastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_module)

    runpy.run_path(server.__file__, run_name="__main__")

    assert DummyFastMCP.last_instance is not None
    assert DummyFastMCP.last_instance.ran is True
