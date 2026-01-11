import sqlite3

import pytest

from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

sqlite_vec = pytest.importorskip("sqlite_vec")

try:
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.close()
except sqlite3.Error:
    pytest.skip("sqlite-vec extension is not loadable", allow_module_level=True)


@pytest.mark.asyncio
async def test_sqlite_vec_store_basic_ops(tmp_path, memory_item):
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        await store.store_memory(memory_item)

        retrieved = await store.get_memory(memory_item.memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == memory_item.memory_id
        assert retrieved.content == memory_item.content

        await store.update_connections(
            memory_item.memory_id, related_ids=["rel-1", "rel-2"], conflict_ids=["conf-1"]
        )
        retrieved_updated = await store.get_memory(memory_item.memory_id)
        assert "rel-1" in retrieved_updated.associations
        assert "rel-2" in retrieved_updated.associations
        assert "conf-1" in retrieved_updated.conflicts_with

        success = await store.delete_memory(memory_item.memory_id)
        assert success is True

        deleted = await store.get_memory(memory_item.memory_id)
        assert deleted is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_metadata_persistence(tmp_path, memory_item):
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    memory_item.associations = ["test-assoc"]
    memory_item.context = {"foo": "bar"}

    try:
        await store.store_memory(memory_item)
    finally:
        await store.close()

    store2 = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store2.initialize()

    try:
        retrieved = await store2.get_memory(memory_item.memory_id)
        assert retrieved.associations == ["test-assoc"]
        assert retrieved.context == {"foo": "bar"}
    finally:
        await store2.close()
