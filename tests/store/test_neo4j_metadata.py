"""Metadata writes use a mocked driver; no Neo4j service is contacted."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("neo4j")

from mnemotree.store.neo4j_store import AsyncGraphDatabase, Neo4jMemoryStore


@pytest.fixture
def metadata_store(monkeypatch):
    session = MagicMock()
    session.run = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = context
    monkeypatch.setattr(AsyncGraphDatabase, "driver", MagicMock(return_value=driver))
    store = Neo4jMemoryStore("neo4j://unused", "unused", "unused")
    # Schema setup is separate; this test exercises the actual metadata write.
    store._initialized = True
    return store, session, driver


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [True, False])
async def test_metadata_write_includes_importance_and_serializes_dates(metadata_store, exists):
    store, session, _ = metadata_store
    result = MagicMock()
    result.single = AsyncMock(return_value={"m": {"importance": 0.75}} if exists else None)
    session.run.return_value = result
    instant = datetime(2024, 1, 2, tzinfo=UTC)
    updated = await store.update_memory_metadata(
        "occurrence-1",
        {
            "importance": 0.75,
            "last_accessed": instant,
            "access_history": [instant],
            "access_count": 2,
            "content": "must not replace source content",
            "unknown": "must not become a property",
        },
    )
    assert updated is exists
    query, params = session.run.await_args.args
    assert "m.importance = $importance" in query
    assert "m.last_accessed = $last_accessed" in query
    assert "m.access_history = $access_history" in query
    assert "m.access_count = $access_count" in query
    assert "content" not in query and "unknown" not in query
    assert params == {
        "memory_id": "occurrence-1",
        "importance": 0.75,
        "last_accessed": instant.isoformat(),
        "access_history": [instant.isoformat()],
        "access_count": 2,
    }
    result.single.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_metadata_only_does_not_write(metadata_store):
    store, session, driver = metadata_store
    assert await store.update_memory_metadata("occurrence-1", {"content": "ignored"}) is False
    session.run.assert_not_awaited()
    driver.session.assert_not_called()
