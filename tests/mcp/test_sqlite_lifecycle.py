"""Exercise MCP mutation and restart against SQLite rather than store mocks."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from langchain_core.embeddings import Embeddings

from mnemotree.core._internal.indexing import BM25Index
from mnemotree.core._internal.scoring import MemoryScorer
from mnemotree.core.memory import MemoryCore
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.scoring import MemoryScoring
from mnemotree.mcp import server
from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore


class LocalTestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [0.0, 1.0] if "zirconium" in text else [1.0, 0.0]


def memory(memory_id, content):
    now = datetime.now(UTC)
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        timestamp=now,
        importance=0.5,
        embedding=[1.0, 0.0],
    )


def test_bm25_replacement_with_empty_tokens_removes_old_statistics():
    index = BM25Index()
    index.add("same", ["old", "old"])
    index.add("same", [])
    assert index.total_len == 0
    assert not index.doc_freq
    assert not index.search(["old"], 10)
    index.remove("same")
    assert not index.term_freqs and not index.doc_len


def test_memory_scorer_orders_before_limiting_and_preserves_ties():
    scorer = MemoryScorer(
        MemoryScoring(
            importance_weight=1, recency_weight=0, relevance_weight=0, score_threshold=0.4
        )
    )
    low, first, second, excluded = [memory(str(i), "content") for i in range(4)]
    low.importance, first.importance, second.importance, excluded.importance = 0.4, 0.9, 0.9, 0.1
    assert scorer.rank([low, first, second, excluded], None) == [first, second, low]


@pytest.mark.asyncio
async def test_sqlite_listing_preserves_occurrences_and_optional_embeddings(tmp_path):
    path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(path)
    try:
        for memory_id in ("first", "second"):
            await store.store_memory(memory(memory_id, "same content"))
        without = await store.list_memories()
        assert {item.memory_id for item in without} == {"first", "second"}
        assert all(item.embedding is None for item in without)
    finally:
        await store.close()
    reopened = SQLiteVecMemoryStore(path)
    try:
        with_vectors = await reopened.list_memories(include_embeddings=True)
        assert len(with_vectors) == 2
        assert all(item.embedding == [1.0, 0.0] for item in with_vectors)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_mcp_sqlite_update_delete_and_restart_are_coherent(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOTREE_MCP_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("MNEMOTREE_MCP_PERSIST_DIR", str(tmp_path / "memories.sqlite"))
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_BM25", "true")
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_NER", "false")
    monkeypatch.setenv("MNEMOTREE_MCP_ENABLE_KEYWORDS", "false")
    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_HOST", raising=False)
    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_PORT", raising=False)
    monkeypatch.setattr(MemoryCore, "_resolve_embeddings", lambda *_: LocalTestEmbeddings())
    monkeypatch.setattr(server, "_memory_core", None)
    core = await server._get_memory_core()
    try:
        await core.persistence.save(memory("first", "obsolete copper"))
        await core.persistence.save(memory("second", "obsolete copper"))
        assert len(core.index_manager.search("obsolete", 10)) == 2

        updated = await server.update_memory("first", {"content": "zirconium revised"})
        assert updated["content"] == "zirconium revised"
        assert [item[0] for item in core.index_manager.search("zirconium", 10)] == ["first"]
        assert [item[0] for item in core.index_manager.search("obsolete", 10)] == ["second"]
        nearest = await core.store.get_similar_memories("zirconium", [0.0, 1.0], top_k=1)
        assert nearest[0].memory_id == "first"
        assert await server.forget("second")
        assert not core.index_manager.search("obsolete", 10)
        assert await core.store.get_memory("second") is None
        accessed = await core.store.get_memory("first")
        await core._update_access_metadata([accessed])
    finally:
        await core.store.close()
        server._memory_core = None

    restarted = await server._get_memory_core()
    try:
        assert [item[0] for item in restarted.index_manager.search("zirconium", 10)] == ["first"]
        assert not restarted.index_manager.search("obsolete", 10)
        cached = restarted.index_manager.get_memory("first")
        assert cached.embedding == [0.0, 1.0]
        assert cached.importance == pytest.approx(0.55)
        assert cached.access_count == 1
        assert await restarted.store.get_memory("second") is None
        before = cached.model_dump()
        # A failed write cannot publish a replacement into the lexical cache.
        monkeypatch.setattr(
            restarted.store, "store_memory", AsyncMock(side_effect=RuntimeError("write"))
        )
        with pytest.raises(RuntimeError, match="write"):
            await server.update_memory("first", {"content": "failed copper"})
        assert restarted.index_manager.get_memory("first").model_dump() == before
    finally:
        await restarted.store.close()
        server._memory_core = None


@pytest.mark.asyncio
async def test_mcp_startup_failure_closes_the_opened_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOTREE_MCP_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("MNEMOTREE_MCP_PERSIST_DIR", str(tmp_path / "memories.sqlite"))
    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_HOST", raising=False)
    monkeypatch.delenv("MNEMOTREE_MCP_CHROMA_PORT", raising=False)
    monkeypatch.setattr(server, "_memory_core", None)
    closed = []
    close = SQLiteVecMemoryStore.close

    async def recording_close(self):
        closed.append(True)
        await close(self)

    def failing_core(**_):
        raise RuntimeError("core")

    monkeypatch.setattr(SQLiteVecMemoryStore, "close", recording_close)
    monkeypatch.setattr(server, "MemoryCore", failing_core)
    with pytest.raises(RuntimeError, match="core"):
        await server._get_memory_core()
    assert closed == [True]
    assert server._memory_core is None
