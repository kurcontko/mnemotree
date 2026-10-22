import asyncio
from datetime import datetime, timezone

import pytest

from mnemotree import Memory, Message, SearchOptions
from mnemotree.evidence import EmbeddingSpec, IngestConflict
from mnemotree.evidence.store import vector_blob


class ToyEmbedder:
    """Deterministic vectors test plumbing, not semantic retrieval quality."""

    spec = EmbeddingSpec(provider="test", model="toy", revision="1", dimensions=2)

    def __init__(self):
        self.calls = 0
        self.fail = False

    async def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider failed")
        return [[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]

    async def embed_query(self, text):
        return (await self.embed_documents([text]))[0]


async def add(memory, text="apple orchard", *, namespace="alice", request="r1", id=None):
    return await memory.add(
        namespace, [Message(content=text, id=id)], request_id=request, session_id="s1"
    )


async def test_persistence_scope_idempotency_and_forgetting(tmp_path):
    path = tmp_path / "memory.db"
    embedder = ToyEmbedder()
    with Memory(path, embedder=embedder) as memory:
        receipt = await add(memory, id="same")
        replay = await add(memory, id="same")
        assert replay.replayed and replay.source_ids == receipt.source_ids
        assert embedder.calls == 1
        with pytest.raises(IngestConflict, match="different payload"):
            await add(memory, "changed")
        await add(memory, namespace="bob", id="same")
    with Memory(path, embedder=embedder) as memory:
        for mode in ("lexical", "dense", "hybrid"):
            result = await memory.search("alice", "apple", options=SearchOptions(mode=mode))
            assert [item.source_id for item in result.evidence] == ["same"]
            assert result.evidence[0].text == "apple orchard"
            assert memory.counter.count(result.context) == result.trace.packed_tokens
        assert memory.source("unknown", "same") is None
        assert not memory.forget("unknown", "same")
        assert memory.forget("alice", "same")
        assert not (await memory.search("alice", "apple")).evidence
        assert (await memory.search("bob", "apple")).evidence
        with pytest.raises(IngestConflict, match="forgotten"):
            await add(memory, id="same")


async def test_durable_atomic_add_and_conflicting_source_ids(tmp_path):
    embedder = ToyEmbedder()
    with Memory(tmp_path / "m.db", embedder=embedder) as memory:
        embedder.fail = True
        with pytest.raises(RuntimeError):
            await add(memory)
        assert memory.store.stats("alice") == {"sources": 0, "chunks": 0, "requests": 0}
        embedder.fail = False
        # An actual SQL failure after a vector was inserted rolls back all indexes.
        memory.store.db.execute(
            "CREATE TRIGGER fail_second BEFORE INSERT ON evidence_chunks "
            "WHEN new.source_id='second' BEGIN SELECT RAISE(ABORT, 'fail'); END"
        )
        with pytest.raises(IngestConflict):
            await memory.add(
                "alice",
                [Message(content="apple", id="first"), Message(content="apple", id="second")],
                request_id="failed",
                session_id="s",
            )
        assert memory.store.stats("alice") == {"sources": 0, "chunks": 0, "requests": 0}
        assert memory.store.db.execute("SELECT count(*) FROM evidence_vec").fetchone()[0] == 0
        assert not (await memory.search("alice", "apple")).evidence
        memory.store.db.execute("DROP TRIGGER fail_second")
        await add(memory, id="first")
        with pytest.raises(IngestConflict):
            await add(memory, request="another", id="first")
        assert memory.store.stats("alice")["sources"] == 1


async def test_occurrences_spans_time_and_neighbor_context(tmp_path):
    content = "😀 café\n" * 10 + "apple orchard\n" + "終わり " * 10
    with Memory(tmp_path / "m.db", chunk_size=40, chunk_overlap=10) as memory:
        receipt = await memory.add(
            "alice",
            [
                Message(content=content, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
                Message(content=content),
            ],
            request_id="r",
            session_id="s",
        )
        assert len(set(receipt.source_ids)) == 2
        result = await memory.search(
            "alice", "apple", options=SearchOptions(mode="lexical", neighbor_chunks=1)
        )
        assert {item.source_id for item in result.evidence} == set(receipt.source_ids)
        for item in result.evidence:
            assert item.text == content[item.start : item.end]
            assert "apple" in item.text and item.session_id == "s"
        assert memory.source("alice", receipt.source_ids[0]).timestamp.year == 2024
        # Search does not create writes, reinforcement, or other retrieval history.
        before = memory.store.db.total_changes
        await memory.search("alice", "apple")
        assert memory.store.db.total_changes == before


async def test_filter_before_cap_and_retrieve_beyond_200(tmp_path):
    with Memory(tmp_path / "m.db", embedder=ToyEmbedder()) as memory:
        await memory.add(
            "bob",
            [Message(content="apple", id=f"b{i}") for i in range(220)],
            request_id="b",
            session_id="b",
        )
        await memory.add(
            "alice",
            [Message(content="banana", id=f"a{i}") for i in range(220)],
            request_id="a",
            session_id="a",
        )
        await add(memory, "apple unique target", request="target", id="target")
        for mode in ("lexical", "dense", "hybrid"):
            result = await memory.search(
                "alice", "apple", options=SearchOptions(mode=mode, limit=1, candidate_limit=1)
            )
            assert [item.source_id for item in result.evidence] == ["target"]


async def test_independent_connections_and_concurrent_retries(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path) as first, Memory(path) as second:
        receipts = await asyncio.gather(add(first), add(second))
        assert sum(receipt.replayed for receipt in receipts) == 1
        assert second.store.stats("alice")["sources"] == 1
        assert (await second.search("alice", 'apple OR "*" - ()')).evidence
        assert not (await second.search("alice", '"*" - ()')).evidence


async def test_reranking_candidate_depth_and_exact_budget(tmp_path):
    class Reverse:
        async def score(self, query, texts):
            return [1 if "short" in text else 0 for text in texts]

    with Memory(tmp_path / "m.db", reranker=Reverse(), chunk_size=5000) as memory:
        await memory.add(
            "alice",
            [
                Message(content="apple " * 600, id="long"),
                Message(content="apple short", id="short"),
            ],
            request_id="r",
            session_id="s",
        )
        result = await memory.search(
            "alice",
            "apple",
            options=SearchOptions(mode="lexical", limit=1, candidate_limit=2, token_budget=200),
        )
        assert len(result.trace.candidates) == 2
        assert result.evidence[0].source_id == "short"
        assert result.trace.packed_tokens <= 200
        assert result.trace.candidates[0].rerank_score == 1
        memory.reranker = None
        result = await memory.search(
            "alice", "apple", options=SearchOptions(mode="lexical", token_budget=200)
        )
        assert [item.source_id for item in result.evidence] == ["short"]
        empty = await memory.search(
            "alice", "apple", options=SearchOptions(mode="lexical", token_budget=0)
        )
        assert not empty.evidence and empty.context == "" and empty.trace.packed_tokens == 0


async def test_embedding_signature_and_lexical_reopen(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path, embedder=ToyEmbedder()) as memory:
        await add(memory)
    wrong = ToyEmbedder()
    wrong.spec = wrong.spec.model_copy(update={"revision": "2"})
    with pytest.raises(ValueError, match="signature mismatch"):
        Memory(path, embedder=wrong)
    with Memory(path) as memory:
        assert (await memory.search("alice", "apple")).evidence
        with pytest.raises(ValueError, match="configured embedder"):
            await add(memory, request="new")
        with pytest.raises(ValueError, match="requires the indexed embedder"):
            await memory.search("alice", "apple", options=SearchOptions(mode="dense"))
    lexical = tmp_path / "lexical.db"
    with Memory(lexical) as memory:
        await add(memory)
    with pytest.raises(ValueError, match="vector rebuild"):
        Memory(lexical, embedder=ToyEmbedder())


@pytest.mark.parametrize(
    "vector", [[0, 0], [1], [float("nan"), 1], [float("inf"), 1], [1e100, 1], [1e-100, 0]]
)
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        vector_blob(vector, 2)


def test_input_validation():
    with pytest.raises(ValueError):
        Message(content="  ")
    with pytest.raises(ValueError):
        Message(content="valid", timestamp=datetime(2024, 1, 1))
    with pytest.raises(ValueError):
        SearchOptions(limit=10, candidate_limit=1)


async def test_invalid_embedding_and_reranker_responses(tmp_path):
    class Broken(ToyEmbedder):
        async def embed_documents(self, texts):
            return [[0.0, 0.0]]

    class BrokenRanker:
        async def score(self, query, texts):
            return [float("nan")]

    with Memory(tmp_path / "bad.db", embedder=Broken()) as memory:
        with pytest.raises(ValueError):
            await add(memory)
        assert memory.store.stats("alice")["requests"] == 0
    with Memory(tmp_path / "rank.db", reranker=BrokenRanker()) as memory:
        await add(memory)
        with pytest.raises(ValueError, match="finite score"):
            await memory.search("alice", "apple")
