from datetime import UTC, datetime

import pytest

from mnemotree import IngestConflict, Memory, MemoryItem, Message, SearchOptions
from mnemotree.evidence import EmbeddingSpec


class ToyEmbedder:
    """Deterministic vectors test plumbing, not semantic retrieval quality."""

    spec = EmbeddingSpec(provider="test", model="toy", revision="1", dimensions=2)

    async def embed_documents(self, texts):
        return [[1.0, 0.0] if "orchid" in text else [0.0, 1.0] for text in texts]

    async def embed_query(self, text):
        return (await self.embed_documents([text]))[0]


async def test_remember_get_recall_forget_round_trip(tmp_path):
    with Memory(tmp_path / "m.db", namespace="alice") as memory:
        when = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
        # Raw text is stored exactly, whitespace and all.
        item = await memory.remember(
            "  I grow orchids\non my balcony. ", role="user", timestamp=when, id="orchids"
        )
        assert item.id == "orchids"
        assert item.content == "  I grow orchids\non my balcony. "
        assert item.role == "user" and item.timestamp == when
        assert await memory.get("orchids") == item

        recall = await memory.recall("orchids")
        assert [found.id for found in recall.items] == ["orchids"]
        assert "orchids" in recall.context
        assert recall.trace.packed_tokens > 0

        assert await memory.forget("orchids")
        assert not await memory.forget("orchids")
        assert await memory.get("orchids") is None
        assert (await memory.recall("orchids")).items == ()


async def test_namespaces_are_isolated_in_one_file(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path, namespace="alice") as alice, Memory(path, namespace="bob") as bob:
        await alice.remember("alice keeps orchids", id="same")
        await bob.remember("bob keeps cacti", id="same")
        assert (await alice.get("same")).content == "alice keeps orchids"
        assert (await bob.get("same")).content == "bob keeps cacti"
        assert (await bob.recall("orchids")).items == ()
        assert await bob.forget("same")
        assert (await alice.get("same")) is not None


async def test_request_id_makes_retries_idempotent(tmp_path):
    with Memory(tmp_path / "m.db", namespace="alice") as memory:
        first = await memory.remember("orchids need humidity", request_id="req-1")
        again = await memory.remember("orchids need humidity", request_id="req-1")
        assert again == first
        with pytest.raises(IngestConflict):
            await memory.remember("something else", request_id="req-1")


async def test_memories_survive_reopening(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path, namespace="alice") as memory:
        stored = await memory.remember("orchids bloom in spring", id="bloom")
    with Memory(path, namespace="alice") as memory:
        assert await memory.get("bloom") == stored
        assert [item.id for item in (await memory.recall("orchids")).items] == ["bloom"]


async def test_remember_many_keeps_turn_order_and_session(tmp_path):
    with Memory(tmp_path / "m.db", namespace="alice") as memory:
        items = await memory.remember_many(
            [
                Message(content="Which plant should I buy?", role="user", id="q"),
                Message(content="An orchid suits a bright balcony.", role="assistant", id="a"),
            ],
            session="chat-1",
        )
        assert [(item.id, item.role, item.session) for item in items] == [
            ("q", "user", "chat-1"),
            ("a", "assistant", "chat-1"),
        ]
        options = SearchOptions(mode="lexical", neighbor_messages=1)
        recall = await memory.recall("orchid balcony", options=options)
        # The neighbouring question comes back as context of the answer it led to.
        assert recall.items[0].id == "a"
        assert "Which plant should I buy?" in recall.context


async def test_separate_remembers_never_become_neighbours(tmp_path):
    with Memory(tmp_path / "m.db", namespace="alice") as memory:
        first = await memory.remember("I bought an orchid", id="one")
        second = await memory.remember("My car is blue", id="two")
        assert first.session != second.session
        options = SearchOptions(mode="lexical", neighbor_messages=2)
        recall = await memory.recall("orchid", options=options)
        assert "car" not in recall.context


async def test_limit_caps_items_and_default_search_applies(tmp_path):
    search = SearchOptions(mode="lexical", limit=1, candidate_limit=10)
    with Memory(tmp_path / "m.db", namespace="alice", search=search) as memory:
        for index in range(3):
            await memory.remember(f"orchid note {index}", id=f"n{index}")
        assert len((await memory.recall("orchid")).items) == 1
        assert len((await memory.recall("orchid", limit=3)).items) == 3


async def test_embedder_enables_hybrid_recall(tmp_path):
    with Memory(tmp_path / "m.db", namespace="alice", embedder=ToyEmbedder()) as memory:
        await memory.remember("orchid care notes", id="orchid")
        await memory.remember("bicycle repair notes", id="bike")
        recall = await memory.recall("orchid")
        assert recall.trace.mode == "hybrid"
        assert recall.items[0].id == "orchid"


def test_namespace_is_required():
    with pytest.raises(TypeError):
        Memory(":memory:")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Memory(":memory:", namespace=" ")


def test_items_are_immutable():
    item = MemoryItem(
        id="x",
        content="text",
        role="user",
        session="s",
        timestamp=None,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError):
        item.content = "changed"  # type: ignore[misc]
