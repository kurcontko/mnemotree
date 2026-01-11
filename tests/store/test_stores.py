import pytest

pytest.importorskip("chromadb")

from mnemotree.store.chromadb_store import ChromaMemoryStore


@pytest.mark.asyncio
async def test_chroma_store_basic_ops(temp_chroma_dir, memory_item):
    store = ChromaMemoryStore(persist_directory=temp_chroma_dir)
    await store.initialize()

    try:
        # Test Store
        await store.store_memory(memory_item)

        # Test Get
        retrieved = await store.get_memory(memory_item.memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == memory_item.memory_id
        assert retrieved.content == memory_item.content

        # Test Update Connections
        await store.update_connections(
            memory_item.memory_id, related_ids=["rel-1", "rel-2"], conflict_ids=["conf-1"]
        )

        retrieved_updated = await store.get_memory(memory_item.memory_id)
        assert "rel-1" in retrieved_updated.associations
        assert "rel-2" in retrieved_updated.associations
        assert "conf-1" in retrieved_updated.conflicts_with

        # Test Delete
        success = await store.delete_memory(memory_item.memory_id)
        assert success is True

        deleted = await store.get_memory(memory_item.memory_id)
        assert deleted is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_chroma_metadata_persistence(temp_chroma_dir, memory_item):
    store = ChromaMemoryStore(persist_directory=temp_chroma_dir)
    await store.initialize()

    memory_item.associations = ["test-assoc"]
    memory_item.context = {"foo": "bar"}

    try:
        await store.store_memory(memory_item)
    finally:
        await store.close()

    # Re-initialize store to ensure persistence works
    store2 = ChromaMemoryStore(persist_directory=temp_chroma_dir)
    await store2.initialize()

    try:
        retrieved = await store2.get_memory(memory_item.memory_id)
        assert retrieved.associations == ["test-assoc"]
        assert retrieved.context == {"foo": "bar"}
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_chroma_graph_index_expansion(temp_chroma_dir, memory_item):
    store = ChromaMemoryStore(persist_directory=temp_chroma_dir)
    await store.initialize()

    try:
        seed = memory_item.model_copy(
            update={
                "memory_id": "graph-1",
                "content": "Paris is in France",
                "entities": {"Paris": "GPE", "France": "GPE"},
            }
        )
        neighbor = memory_item.model_copy(
            update={
                "memory_id": "graph-2",
                "content": "France has many vineyards",
                "entities": {"France": "GPE"},
            }
        )

        await store.store_memory(seed)
        await store.store_memory(neighbor)

        results = await store.query_by_entities(["paris"])
        ids = {memory.memory_id for memory in results}
        assert "graph-1" in ids
        assert "graph-2" in ids
    finally:
        await store.close()
