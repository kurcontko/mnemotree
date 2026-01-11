import pytest

pytest.importorskip("chromadb")

from mnemotree.core.query import MemoryQuery
from mnemotree.store.chromadb_store import ChromaMemoryStore


@pytest.mark.asyncio
async def test_store_contract_roundtrip_and_query_limit(temp_chroma_dir, memory_item):
    store = ChromaMemoryStore(persist_directory=temp_chroma_dir)
    await store.initialize()

    try:
        first = memory_item.model_copy(update={"memory_id": "contract-1", "content": "Contract one"})
        second = memory_item.model_copy(update={"memory_id": "contract-2", "content": "Contract two"})
        third = memory_item.model_copy(update={"memory_id": "contract-3", "content": "Contract three"})

        await store.store_memory(first)
        await store.store_memory(second)
        await store.store_memory(third)

        retrieved = await store.get_memory(first.memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == first.memory_id
        assert retrieved.content == first.content
        assert retrieved.memory_type == first.memory_type
        assert retrieved.importance == first.importance

        query = MemoryQuery(vector=first.embedding, limit=2)
        results = await store.query_memories(query)
        assert len(results) <= 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_contract_metadata_update(temp_chroma_dir, memory_item):
    store = ChromaMemoryStore(persist_directory=temp_chroma_dir)
    await store.initialize()

    try:
        memory = memory_item.model_copy(update={"memory_id": "contract-update"})
        await store.store_memory(memory)

        success = await store.update_memory_metadata(
            memory.memory_id,
            {"tags": ["alpha"], "context": {"foo": "bar"}},
        )
        assert success is True

        updated = await store.get_memory(memory.memory_id)
        assert updated is not None
        assert updated.tags == ["alpha"]
        assert updated.context == {"foo": "bar"}
    finally:
        await store.close()
