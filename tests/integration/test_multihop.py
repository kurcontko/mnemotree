"""End-to-end multi-hop retrieval tests using an in-memory SQLiteVecMemoryStore.

These tests verify the full pipeline: store memories → create links → run PPR / SCMRAG.
They skip if sqlite_vec is unavailable (CI without the optional dep).
"""
from __future__ import annotations

import pytest

from mnemotree.core.models import LinkType, MemoryItem, MemoryType
from mnemotree.core.ppr import PPRConfig, build_adjacency_from_links, personalized_pagerank
from mnemotree.core.query import TypedPathQuery
from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

pytest_plugins = ["anyio"]

try:
    import sqlite_vec  # noqa: F401

    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

pytestmark = pytest.mark.skipif(
    not _HAS_SQLITE_VEC, reason="sqlite_vec not installed"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path):
    s = SQLiteVecMemoryStore(db_path=":memory:", collection_name="test_multihop")
    await s.initialize()
    yield s
    await s.close()


def _memory(mid: str, content: str, embedding: list[float] | None = None) -> MemoryItem:
    emb = embedding or ([0.1] * 16)
    return MemoryItem(
        memory_id=mid,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=emb,
    )


# ---------------------------------------------------------------------------
# get_neighborhood_links
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_neighborhood_links_depth1(store):
    await store.store_memory(_memory("A", "Alice"))
    await store.store_memory(_memory("B", "Bob"))
    await store.store_memory(_memory("C", "Carol"))
    await store.create_link("A", "B", LinkType.REFERENCES)
    await store.create_link("B", "C", LinkType.REFERENCES)

    links = await store.get_neighborhood_links(["A"], max_depth=1)
    target_ids = {lk.target_id for lk in links}
    assert "B" in target_ids
    assert "C" not in target_ids  # depth 1 only


@pytest.mark.asyncio
async def test_get_neighborhood_links_depth2(store):
    await store.store_memory(_memory("A", "Alice"))
    await store.store_memory(_memory("B", "Bob"))
    await store.store_memory(_memory("C", "Carol"))
    await store.create_link("A", "B", LinkType.REFERENCES)
    await store.create_link("B", "C", LinkType.REFERENCES)

    links = await store.get_neighborhood_links(["A"], max_depth=2)
    target_ids = {lk.target_id for lk in links}
    assert "B" in target_ids
    assert "C" in target_ids


@pytest.mark.asyncio
async def test_get_neighborhood_links_link_type_filter(store):
    await store.store_memory(_memory("A", "Alice"))
    await store.store_memory(_memory("B", "Bob"))
    await store.store_memory(_memory("C", "Carol"))
    await store.create_link("A", "B", LinkType.CAUSES)
    await store.create_link("A", "C", LinkType.PART_OF)

    links = await store.get_neighborhood_links(
        ["A"], max_depth=1, link_types=[LinkType.CAUSES]
    )
    target_ids = {lk.target_id for lk in links}
    assert "B" in target_ids
    assert "C" not in target_ids


@pytest.mark.asyncio
async def test_get_neighborhood_links_multiple_seeds(store):
    for mid in ["A", "B", "C", "D"]:
        await store.store_memory(_memory(mid, f"Memory {mid}"))
    await store.create_link("A", "C", LinkType.REFERENCES)
    await store.create_link("B", "D", LinkType.REFERENCES)

    links = await store.get_neighborhood_links(["A", "B"], max_depth=1)
    target_ids = {lk.target_id for lk in links}
    assert "C" in target_ids
    assert "D" in target_ids


# ---------------------------------------------------------------------------
# traverse_typed_path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_typed_path_single_hop(store):
    await store.store_memory(_memory("A", "Cause A"))
    await store.store_memory(_memory("B", "Effect B"))
    await store.create_link("A", "B", LinkType.CAUSES)

    results = await store.traverse_typed_path("A", [LinkType.CAUSES])
    assert len(results) == 1
    end_memory, path_links = results[0]
    assert end_memory.memory_id == "B"
    assert len(path_links) == 1
    assert path_links[0].link_type == LinkType.CAUSES


@pytest.mark.asyncio
async def test_traverse_typed_path_two_hops(store):
    for mid in ["A", "B", "C"]:
        await store.store_memory(_memory(mid, f"Memory {mid}"))
    await store.create_link("A", "B", LinkType.CAUSES)
    await store.create_link("B", "C", LinkType.PART_OF)

    results = await store.traverse_typed_path("A", [LinkType.CAUSES, LinkType.PART_OF])
    assert len(results) == 1
    end_memory, path_links = results[0]
    assert end_memory.memory_id == "C"
    assert len(path_links) == 2


@pytest.mark.asyncio
async def test_traverse_typed_path_wrong_type_returns_empty(store):
    await store.store_memory(_memory("A", "Alice"))
    await store.store_memory(_memory("B", "Bob"))
    await store.create_link("A", "B", LinkType.SUPPORTS)

    # Ask for CAUSES but link is SUPPORTS
    results = await store.traverse_typed_path("A", [LinkType.CAUSES])
    assert results == []


@pytest.mark.asyncio
async def test_traverse_typed_path_empty_pattern(store):
    await store.store_memory(_memory("A", "Alice"))
    results = await store.traverse_typed_path("A", [])
    assert results == []


# ---------------------------------------------------------------------------
# PPR integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ppr_finds_two_hop_memory(store):
    """A → B → C: C should get a non-zero PPR score when seeding A."""
    for mid in ["A", "B", "C"]:
        await store.store_memory(_memory(mid, f"Memory {mid}"))
    await store.create_link("A", "B", LinkType.REFERENCES, strength=0.9)
    await store.create_link("B", "C", LinkType.REFERENCES, strength=0.9)

    links = await store.get_neighborhood_links(["A"], max_depth=2)
    adj = build_adjacency_from_links(links, PPRConfig())
    scores = personalized_pagerank({"A": 1.0}, adj, PPRConfig(alpha=0.15, max_iter=100))

    assert scores.get("C", 0.0) > 0.0


# ---------------------------------------------------------------------------
# TypedPathQuery dataclass
# ---------------------------------------------------------------------------


def test_typed_path_query_defaults():
    q = TypedPathQuery()
    assert q.start_ids == []
    assert q.path_pattern == []
    assert q.min_strength == 0.0
    assert q.limit == 10


def test_typed_path_query_with_values():
    q = TypedPathQuery(
        start_ids=["m1"],
        path_pattern=[LinkType.CAUSES, LinkType.PART_OF],
        min_strength=0.5,
        seed_query="find causes",
        limit=5,
    )
    assert q.path_pattern == [LinkType.CAUSES, LinkType.PART_OF]
    assert q.limit == 5
