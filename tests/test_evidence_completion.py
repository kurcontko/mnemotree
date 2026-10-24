"""Long-source completion must not reread whole documents for every chunk."""

import pytest

from mnemotree import Memory, Message, SearchOptions


def completion_options():
    return SearchOptions(
        mode="lexical",
        limit=100,
        candidate_limit=100,
        deep_candidate_limit=300,
        deep_history_density=2.0,
        token_budget=512,
        chunks_per_source=1,
        neighbor_messages=1,
        merge_blocks=True,
        role_first="user",
        role_first_min_assistant_share=0.65,
    )


@pytest.mark.parametrize("length", [20_000, 80_000])
async def test_completion_reads_each_source_once_and_preserves_output(tmp_path, monkeypatch, length):
    with Memory(tmp_path / "memory.db") as memory:
        text = "needle " + "ordinary words " * (length // 15)
        await memory.add(
            "alice", [Message(content=text, id="long")], request_id="r", session_id="s"
        )
        original_get = memory.store.get_source
        reads = []

        def counted_get(namespace, source_id):
            source = original_get(namespace, source_id)
            reads.append(len(source.content) if source else 0)
            return source

        monkeypatch.setattr(memory.store, "get_source", counted_get)
        result = await memory.search("alice", "needle", options=completion_options())
        assert reads == [len(text)]
        assert result.trace.packed_tokens <= 512

        # Simulate the previous per-window source reads using the same retrieval,
        # token counter, source spans, and packing order.
        original_window = memory.store.window

        def uncached_window(namespace, chunk, neighbors, *, source=None):
            return original_window(namespace, chunk, neighbors)

        monkeypatch.setattr(memory.store, "window", uncached_window)
        previous = await memory.search("alice", "needle", options=completion_options())
        assert len(reads) > 2
        assert result.context == previous.context
        assert result.evidence == previous.evidence
        assert result.trace.packed_tokens == previous.trace.packed_tokens
        assert result.trace.omitted_ids == previous.trace.omitted_ids


async def test_window_rejects_cached_source_from_other_scope(tmp_path):
    with Memory(tmp_path / "memory.db") as memory:
        for namespace in ("alice", "bob"):
            await memory.add(
                namespace,
                [Message(content="needle", id="same")],
                request_id="r",
                session_id="s",
            )
        chunk = memory.store.lexical("alice", "needle", 1)[0][0]
        other = memory.store.get_source("bob", "same")
        with pytest.raises(ValueError, match="namespace and ID"):
            memory.store.window("alice", chunk, 0, source=other)
