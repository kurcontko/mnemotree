import sqlite3
from datetime import datetime, timezone

import pytest

from mnemotree.core.models import LinkType, MemoryItem, MemoryType
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

EMBEDDING_DIM = 8


def _make_memory(memory_id: str, content: str, embedding: list[float] | None = None) -> MemoryItem:
    if embedding is None:
        # Generate a simple deterministic embedding based on id
        base = hash(memory_id) % 100 / 100.0
        embedding = [base + i * 0.01 for i in range(EMBEDDING_DIM)]
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp=str(datetime.now(timezone.utc)),
        embedding=embedding,
    )


@pytest.fixture
async def store(tmp_path):
    db_path = tmp_path / "test_kg.sqlite"
    s = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=EMBEDDING_DIM)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def populated_store(store):
    """Store with 3 memories: m1, m2, m3."""
    m1 = _make_memory("m1", "First memory about Python")
    m2 = _make_memory("m2", "Second memory about testing")
    m3 = _make_memory("m3", "Third memory about databases")
    await store.store_memory(m1)
    await store.store_memory(m2)
    await store.store_memory(m3)
    return store


@pytest.mark.asyncio
async def test_create_link(populated_store):
    store = populated_store
    link = await store.create_link(
        "m1",
        "m2",
        LinkType.REFERENCES,
        context="m1 references m2",
    )
    assert link.source_id == "m1"
    assert link.target_id == "m2"
    assert link.link_type == LinkType.REFERENCES
    assert link.context == "m1 references m2"
    assert link.strength == 1.0
    assert link.link_id  # non-empty


@pytest.mark.asyncio
async def test_create_bidirectional_link(populated_store):
    store = populated_store
    await store.create_link(
        "m1",
        "m2",
        LinkType.SIMILAR_TO,
        bidirectional=True,
    )
    # Should have the forward link
    outgoing = await store.get_links("m1", direction="outgoing")
    assert any(lk.target_id == "m2" and lk.link_type == LinkType.SIMILAR_TO for lk in outgoing)

    # Should have the reverse link
    outgoing_m2 = await store.get_links("m2", direction="outgoing")
    assert any(lk.target_id == "m1" and lk.link_type == LinkType.SIMILAR_TO for lk in outgoing_m2)


@pytest.mark.asyncio
async def test_get_links_direction_filter(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m3", "m1", LinkType.SUPPORTS)

    outgoing = await store.get_links("m1", direction="outgoing")
    assert len(outgoing) == 1
    assert outgoing[0].target_id == "m2"

    incoming = await store.get_links("m1", direction="incoming")
    assert len(incoming) == 1
    assert incoming[0].source_id == "m3"

    both = await store.get_links("m1", direction="both")
    assert len(both) == 2


@pytest.mark.asyncio
async def test_get_links_type_filter(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m1", "m3", LinkType.SUPPORTS)

    refs_only = await store.get_links("m1", link_types=[LinkType.REFERENCES])
    assert len(refs_only) == 1
    assert refs_only[0].link_type == LinkType.REFERENCES

    both_types = await store.get_links("m1", link_types=[LinkType.REFERENCES, LinkType.SUPPORTS])
    assert len(both_types) == 2


@pytest.mark.asyncio
async def test_get_links_strength_filter(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES, strength=0.3)
    await store.create_link("m1", "m3", LinkType.SUPPORTS, strength=0.8)

    strong_links = await store.get_links("m1", min_strength=0.5)
    assert len(strong_links) == 1
    assert strong_links[0].strength == 0.8


@pytest.mark.asyncio
async def test_get_backlinks(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m3", "m2", LinkType.SUPPORTS)

    backlinks = await store.get_backlinks("m2")
    assert len(backlinks) == 2
    backlink_ids = {m.memory_id for m in backlinks}
    assert backlink_ids == {"m1", "m3"}


@pytest.mark.asyncio
async def test_get_backlinks_type_filter(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m3", "m2", LinkType.SUPPORTS)

    refs_only = await store.get_backlinks("m2", link_types=[LinkType.REFERENCES])
    assert len(refs_only) == 1
    assert refs_only[0].memory_id == "m1"


@pytest.mark.asyncio
async def test_traverse_bfs(populated_store):
    store = populated_store
    # m1 -> m2 -> m3
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m2", "m3", LinkType.ELABORATES)

    result = await store.traverse_graph("m1", max_depth=3, strategy="bfs")
    assert len(result) == 2

    # First hop should be m2
    assert result[0][0].memory_id == "m2"
    assert result[0][1] == 1  # depth
    assert len(result[0][2]) == 1  # one link in path

    # Second hop should be m3
    assert result[1][0].memory_id == "m3"
    assert result[1][1] == 2  # depth
    assert len(result[1][2]) == 2  # two links in path


@pytest.mark.asyncio
async def test_traverse_dfs(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m2", "m3", LinkType.ELABORATES)

    result = await store.traverse_graph("m1", max_depth=3, strategy="dfs")
    # Should still find both nodes
    found_ids = {r[0].memory_id for r in result}
    assert found_ids == {"m2", "m3"}


@pytest.mark.asyncio
async def test_traverse_link_type_filter(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m1", "m3", LinkType.SUPPORTS)

    result = await store.traverse_graph("m1", max_depth=2, link_types=[LinkType.REFERENCES])
    assert len(result) == 1
    assert result[0][0].memory_id == "m2"


@pytest.mark.asyncio
async def test_find_path(populated_store):
    store = populated_store
    # m1 -> m2 -> m3
    await store.create_link("m1", "m2", LinkType.REFERENCES)
    await store.create_link("m2", "m3", LinkType.ELABORATES)

    path = await store.find_path("m1", "m3")
    assert path is not None
    assert len(path) == 2

    # First step: m2 via the m1->m2 link
    assert path[0][0].memory_id == "m2"
    assert path[0][1].link_type == LinkType.REFERENCES

    # Second step: m3 via the m2->m3 link
    assert path[1][0].memory_id == "m3"
    assert path[1][1].link_type == LinkType.ELABORATES


@pytest.mark.asyncio
async def test_find_path_no_connection(populated_store):
    store = populated_store
    # No links at all
    path = await store.find_path("m1", "m3")
    assert path is None


@pytest.mark.asyncio
async def test_find_path_direct(populated_store):
    store = populated_store
    await store.create_link("m1", "m2", LinkType.REFERENCES)

    path = await store.find_path("m1", "m2")
    assert path is not None
    assert len(path) == 1
    assert path[0][0].memory_id == "m2"


@pytest.mark.asyncio
async def test_suggest_links(tmp_path):
    """Test that suggest_links returns candidates based on embedding similarity."""
    db_path = tmp_path / "test_suggest.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=EMBEDDING_DIM)
    await store.initialize()

    try:
        # Create memories with similar embeddings
        base_embed = [0.5] * EMBEDDING_DIM
        similar_embed = [0.51] * EMBEDDING_DIM  # very close
        different_embed = [0.0] * EMBEDDING_DIM  # far away

        m1 = _make_memory("m1", "Base memory", embedding=base_embed)
        m2 = _make_memory("m2", "Similar memory", embedding=similar_embed)
        m3 = _make_memory("m3", "Different memory", embedding=different_embed)

        await store.store_memory(m1)
        await store.store_memory(m2)
        await store.store_memory(m3)

        suggestions = await store.suggest_links("m1", limit=5, min_similarity=0.5)

        # m2 should be suggested (close embedding), m3 might not be (far away)
        suggested_ids = {s[0].memory_id for s in suggestions}
        assert "m2" in suggested_ids
        # All suggestions should have SIMILAR_TO type
        for _, link_type, confidence, _ in suggestions:
            assert link_type == LinkType.SIMILAR_TO
            assert 0.0 <= confidence <= 1.0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_suggest_links_excludes_linked(tmp_path):
    """Already-linked memories should be excluded from suggestions."""
    db_path = tmp_path / "test_suggest_excl.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=EMBEDDING_DIM)
    await store.initialize()

    try:
        embed = [0.5] * EMBEDDING_DIM
        m1 = _make_memory("m1", "Memory A", embedding=embed)
        m2 = _make_memory("m2", "Memory B", embedding=[0.51] * EMBEDDING_DIM)
        await store.store_memory(m1)
        await store.store_memory(m2)

        # Link m1 to m2
        await store.create_link("m1", "m2", LinkType.REFERENCES)

        # m2 should now be excluded from suggestions
        suggestions = await store.suggest_links("m1", min_similarity=0.0)
        suggested_ids = {s[0].memory_id for s in suggestions}
        assert "m2" not in suggested_ids
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_update_link_strength(populated_store):
    store = populated_store
    link = await store.create_link("m1", "m2", LinkType.REFERENCES, strength=0.5)

    success = await store.update_link_strength(link.link_id, 0.9)
    assert success is True

    # Verify the update
    links = await store.get_links("m1", direction="outgoing")
    assert len(links) == 1
    assert links[0].strength == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_update_link_strength_nonexistent(populated_store):
    store = populated_store
    success = await store.update_link_strength("nonexistent-link-id", 0.5)
    assert success is False


@pytest.mark.asyncio
async def test_delete_link(populated_store):
    store = populated_store
    link = await store.create_link("m1", "m2", LinkType.REFERENCES)

    success = await store.delete_link(link.link_id)
    assert success is True

    # Verify deletion
    links = await store.get_links("m1", direction="outgoing")
    assert len(links) == 0


@pytest.mark.asyncio
async def test_delete_link_nonexistent(populated_store):
    store = populated_store
    success = await store.delete_link("nonexistent-link-id")
    assert success is False


@pytest.mark.asyncio
async def test_protocol_compliance(populated_store):
    """Verify that SQLiteVecMemoryStore satisfies SupportsKnowledgeGraph protocol."""
    from mnemotree.store.protocols import SupportsKnowledgeGraph

    assert isinstance(populated_store, SupportsKnowledgeGraph)
