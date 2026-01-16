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


def test_parse_memory_type():
    assert server._parse_memory_type(None) is None
    assert server._parse_memory_type(" semantic ") == MemoryType.SEMANTIC
    assert server._parse_memory_type("EPISODIC") == MemoryType.EPISODIC
    with pytest.raises(ValueError, match="Unknown memory_type"):
        server._parse_memory_type("unknown")


def test_serialize_memory_excludes_embedding():
    memory = _make_memory(
        "mem-1",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        embedding=[0.1, 0.2],
    )
    data = server._serialize_memory(memory, include_embedding=False)
    assert data["memory_id"] == "mem-1"
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
    assert len(instance.tools) == 9


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
    assert core.kwargs["ner"] == "ner-instance"
    assert core.kwargs["enable_ner"] is True
    assert core.kwargs["enable_keywords"] is True


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
    assert core.kwargs["ner"] is None


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
async def test_remember_recall_search_index(monkeypatch):
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
        analyze=True,
        summarize=False,
        references=["ref"],
        include_embedding=False,
    )

    assert stored["memory_id"] == "mem-remember"
    assert "embedding" not in stored
    memory_core.remember.assert_awaited_once_with(
        content="hello",
        memory_type=MemoryType.EPISODIC,
        importance=0.9,
        tags=["tag"],
        context={"source": "test"},
        analyze=True,
        summarize=False,
        references=["ref"],
    )

    recalled = await server.recall(
        query="hello",
        limit=5,
        scoring=False,
        update_access=True,
        include_embedding=True,
    )
    assert recalled[0]["embedding"] == [0.3]
    memory_core.recall.assert_awaited_once_with(
        query="hello",
        limit=5,
        scoring=False,
        update_access=True,
    )

    results = await server.search_index(query="hello", limit=3, include_summary=False)
    assert "summary" not in results[0]


@pytest.mark.asyncio
async def test_timeline_empty_memories(monkeypatch):
    async def _fake_get_all_memories(_, *, include_embeddings):
        return []

    monkeypatch.setattr(server, "_get_all_memories", _fake_get_all_memories)
    monkeypatch.setattr(server, "_get_memory_core", AsyncMock())

    assert await server.timeline(memory_id="mem-1") == []


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

    results = await server.get_memories(["mem-keep", "mem-missing"], include_embedding=True)

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
    assert await server.memory_types() == [item.value for item in MemoryType]
    assert await server.health() == {"status": "ok"}


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
