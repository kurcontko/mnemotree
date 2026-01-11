from __future__ import annotations

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.store.sqlite_graph import SQLiteGraphIndex


def _memory(
    memory_id: str,
    *,
    entities: dict[str, str],
    entity_mentions: dict[str, list[str]] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=f"content-{memory_id}",
        memory_type=MemoryType.EPISODIC,
        importance=0.5,
        embedding=[0.0],
        entities=entities,
        entity_mentions=entity_mentions or {},
    )


def test_graph_index_hops_and_delete(tmp_path) -> None:
    index = SQLiteGraphIndex(str(tmp_path / "graph.sqlite3"))

    m1 = _memory("m1", entities={"john": "person", "acme": "org"})
    m2 = _memory("m2", entities={"john": "person", "projectx": "project"})
    m3 = _memory("m3", entities={"projectx": "project", "kubernetes": "tech"})

    index.upsert_memory(m1)
    index.upsert_memory(m2)
    index.upsert_memory(m3)

    hits = index.recall_by_entities(["john"], limit=10, hops=2)
    assert {hit.memory_id for hit in hits} == {"m1", "m2", "m3"}
    assert any(hit.memory_id == "m3" and hit.depth == 2 for hit in hits)

    index.delete_memory("m2")
    hits_after_delete = index.recall_by_entities(["john"], limit=10, hops=2)
    assert {hit.memory_id for hit in hits_after_delete} == {"m1"}


def test_graph_index_mention_counts_affect_score(tmp_path) -> None:
    index = SQLiteGraphIndex(str(tmp_path / "graph.sqlite3"))

    m1 = _memory(
        "m1",
        entities={"john": "person"},
        entity_mentions={"john": ["a", "b", "c"]},
    )
    m2 = _memory(
        "m2",
        entities={"john": "person"},
        entity_mentions={"john": ["a"]},
    )

    index.upsert_memory(m1)
    index.upsert_memory(m2)

    hits = index.recall_by_entities(["john"], limit=10, hops=1)
    assert [hit.memory_id for hit in hits] == ["m1", "m2"]
