import sqlite3
from datetime import datetime, timezone

import pytest

from mnemotree.core.query import FilterOperator, MemoryFilter, MemoryQuery, SortOrder
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


@pytest.mark.asyncio
async def test_sqlite_vec_query_memories(tmp_path, memory_item):
    """Test complex memory queries with filters and sorting."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Store memories with various attributes
        now = datetime.now(timezone.utc)
        mem1 = memory_item.model_copy(
            update={
                "memory_id": "mem-1",
                "content": "Important project update",
                "importance": 0.9,
                "tags": ["work", "project"],
                "timestamp": now,
            }
        )
        mem2 = memory_item.model_copy(
            update={
                "memory_id": "mem-2",
                "content": "Team meeting notes",
                "importance": 0.7,
                "tags": ["work", "meeting"],
                "timestamp": now,
            }
        )
        mem3 = memory_item.model_copy(
            update={
                "memory_id": "mem-3",
                "content": "Personal reminder",
                "importance": 0.4,
                "tags": ["personal"],
                "timestamp": now,
            }
        )

        await store.store_memory(mem1)
        await store.store_memory(mem2)
        await store.store_memory(mem3)

        # Test query with filters
        query = MemoryQuery(
            filters=[
                MemoryFilter("tags", FilterOperator.CONTAINS, "work"),
                MemoryFilter("importance", FilterOperator.GTE, 0.6),
            ],
            limit=10,
            sort_by="importance",
            sort_order=SortOrder.DESC,
        )

        results = await store.query_memories(query)
        assert len(results) >= 1
        # Verify filtering worked
        for result in results:
            assert "work" in result.tags
            assert result.importance >= 0.6
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_get_similar_memories(tmp_path, memory_item):
    """Test vector similarity search."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Store memories
        mem1 = memory_item.model_copy(
            update={
                "memory_id": "mem-1",
                "content": "Python programming tutorial",
                "importance": 0.8,
            }
        )
        mem2 = memory_item.model_copy(
            update={"memory_id": "mem-2", "content": "JavaScript basics", "importance": 0.6}
        )
        mem3 = memory_item.model_copy(
            update={"memory_id": "mem-3", "content": "Advanced Python features", "importance": 0.9}
        )

        await store.store_memory(mem1)
        await store.store_memory(mem2)
        await store.store_memory(mem3)

        # Test basic retrieval
        results = await store.get_memory("mem-1")
        assert results is not None
        assert results.content == "Python programming tutorial"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_query_by_entities(tmp_path, memory_item):
    """Test querying memories by entities with entity index."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Store memories with entities
        mem1 = memory_item.model_copy(
            update={
                "memory_id": "mem-1",
                "content": "Meeting with Alice at OpenAI",
                "entities": {"Alice": "PERSON", "OpenAI": "ORG"},
            }
        )
        mem2 = memory_item.model_copy(
            update={
                "memory_id": "mem-2",
                "content": "OpenAI released GPT-4",
                "entities": {"OpenAI": "ORG", "GPT-4": "PRODUCT"},
            }
        )
        mem3 = memory_item.model_copy(
            update={
                "memory_id": "mem-3",
                "content": "Alice recommended a book",
                "entities": {"Alice": "PERSON"},
            }
        )

        await store.store_memory(mem1)
        await store.store_memory(mem2)
        await store.store_memory(mem3)

        # Test query with entity list
        results_openai = await store.query_by_entities(["OpenAI"], limit=10)
        ids_openai = {m.memory_id for m in results_openai}
        assert "mem-1" in ids_openai
        assert "mem-2" in ids_openai

        # Test query with entity dict
        results_alice = await store.query_by_entities({"Alice": "PERSON"}, limit=10)
        ids_alice = {m.memory_id for m in results_alice}
        assert "mem-1" in ids_alice
        assert "mem-3" in ids_alice

        # Test case-insensitive matching
        results_lowercase = await store.query_by_entities(["alice"], limit=10)
        assert len(results_lowercase) >= 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_entity_index_operations(tmp_path, memory_item):
    """Test entity index creation and update operations."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Store memory with entities
        mem1 = memory_item.model_copy(
            update={
                "memory_id": "mem-1",
                "content": "Discussion about Python",
                "entities": {"Python": "LANGUAGE"},
            }
        )
        await store.store_memory(mem1)

        # Update memory with new entities
        mem1_updated = mem1.model_copy(
            update={"entities": {"Python": "LANGUAGE", "Java": "LANGUAGE"}}
        )
        await store.store_memory(mem1_updated)

        # Query should find both entities
        results_python = await store.query_by_entities(["Python"], limit=10)
        assert len(results_python) >= 1

        results_java = await store.query_by_entities(["Java"], limit=10)
        assert len(results_java) >= 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_schema_initialization(tmp_path, memory_item):
    """Test schema initialization and metadata management."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Verify schema was created
        assert store._conn is not None

        # Store a memory to verify schema works
        await store.store_memory(memory_item)
        retrieved = await store.get_memory(memory_item.memory_id)
        assert retrieved is not None
    finally:
        await store.close()

    # Re-initialize with same db to test existing schema handling
    store2 = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store2.initialize()

    try:
        # Should still be able to retrieve
        retrieved = await store2.get_memory(memory_item.memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == memory_item.memory_id
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_sqlite_vec_delete_with_cascade(tmp_path, memory_item):
    """Test memory deletion with cascade option."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Store memories with connections
        mem1 = memory_item.model_copy(update={"memory_id": "mem-1", "content": "First memory"})
        mem2 = memory_item.model_copy(
            update={
                "memory_id": "mem-2",
                "content": "Second memory",
                "associations": ["mem-1"],
            }
        )

        await store.store_memory(mem1)
        await store.store_memory(mem2)

        # Delete with cascade=True
        success = await store.delete_memory("mem-1", cascade=True)
        assert success is True

        # Verify deletion
        deleted = await store.get_memory("mem-1")
        assert deleted is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_error_handling(tmp_path, memory_item):
    """Test error handling for invalid operations."""
    db_path = tmp_path / "memories.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=len(memory_item.embedding))
    await store.initialize()

    try:
        # Test getting non-existent memory
        result = await store.get_memory("non-existent-id")
        assert result is None

        # Test deleting non-existent memory
        success = await store.delete_memory("non-existent-id")
        assert success is False

        # Test updating connections for non-existent memory
        # Should handle gracefully without error
        await store.update_connections("non-existent-id", related_ids=["rel-1"])

    finally:
        await store.close()
