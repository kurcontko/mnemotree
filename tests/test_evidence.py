import asyncio
import random
from datetime import UTC, datetime, timedelta, timezone
from importlib import import_module

import pytest

from mnemotree.evidence import (
    EmbeddingSpec,
    IngestConflict,
    IngestReceipt,
    Memory,
    Message,
    SearchOptions,
)
from mnemotree.evidence.engine import QUERY_EMBED_CHARS, QUERY_EMBED_TOKENS, fuse
from mnemotree.evidence.models import Chunk
from mnemotree.evidence.packing import LINE_SAFE_ENCODINGS, TiktokenCounter
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
            assert result.trace.packed_tokens == memory.counter.count(result.context)
            assert result.trace.evidence_tokens == sum(
                memory.counter.count(item.render()) for item in result.evidence
            )
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
                Message(content=content, timestamp=datetime(2024, 1, 1, tzinfo=UTC)),
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
        receipts = await asyncio.gather(
            asyncio.to_thread(asyncio.run, add(first)),
            asyncio.to_thread(asyncio.run, add(second)),
        )
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


@pytest.mark.parametrize("order", ["rank", "time"])
async def test_context_budget_includes_separators_in_presentation_order(tmp_path, order):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "alice",
            [
                Message(id="red", content="orchid red", timestamp=None),
                Message(
                    id="blue", content="orchid blue", timestamp=datetime(2024, 1, 1, tzinfo=UTC)
                ),
            ],
            request_id="r",
            session_id="s",
        )
        settings = SearchOptions(mode="lexical", order=order)
        full = await memory.search("alice", "orchid", options=settings)
        encoding = memory.counter.encoding
        exact = len(encoding.encode(full.context))
        assert len(full.evidence) == 2
        assert full.trace.packed_tokens == exact
        assert full.trace.evidence_tokens < exact
        ranked = await memory.search("alice", "orchid", options=SearchOptions(mode="lexical"))
        if order == "time":
            # Dated items come first, undated ones after.
            assert [item.source_id for item in full.evidence] == ["blue", "red"]
        else:
            assert full.evidence == ranked.evidence
        tight = await memory.search(
            "alice", "orchid", options=settings.model_copy(update={"token_budget": exact - 1})
        )
        assert len(tight.evidence) == 1
        # Presentation order never decides what is kept.
        assert tight.evidence[0].source_id == ranked.evidence[0].source_id
        assert tight.trace.packed_tokens == len(encoding.encode(tight.context)) <= exact - 1
        assert tight.trace.evidence_tokens <= exact - 1
        assert len(tight.sources) == 1
        assert "sources" not in tight.model_dump()


async def test_joined_budget_holds_through_merges_and_neighbors(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "u",
            [
                Message(id="a0", content="Boston marathon plans for spring"),
                Message(id="a1", content="Boston marathon finish was 3 hours"),
                Message(id="a2", content="Thanks for the pacing advice"),
            ],
            request_id="a",
            session_id="a",
        )
        await memory.add(
            "u",
            [
                Message(id="b0", content="Boston again next year"),
                Message(id="b1", content="That sounds like a fun trip"),
            ],
            request_id="b",
            session_id="b",
        )
        # No closing punctuation: ".\n\n" is one token, which would hide the separator cost.
        settings = SearchOptions(mode="lexical", neighbor_messages=1, merge_blocks=True)
        full = await memory.search("u", "Boston", options=settings)
        assert len(full.evidence) == 2
        assert any(item.merged for item in full.evidence)
        assert any(member.neighbors for item in full.evidence for member in item.members())

        contexts = set()
        for budget in range(full.trace.packed_tokens + 1):
            result = await memory.search(
                "u", "Boston", options=settings.model_copy(update={"token_budget": budget})
            )
            assert result.trace.packed_tokens == memory.counter.count(result.context) <= budget
            assert result.trace.evidence_tokens <= budget
            contexts.add(result.context)
        assert result.context == full.context
        # The sweep reaches partial packings, not only empty and complete ones.
        assert len(contexts) > 2


async def test_sources_list_a_message_once_across_its_chunks(tmp_path):
    content = "apple orchard harvest notes\n" * 12
    with Memory(tmp_path / "m.db", chunk_size=40, chunk_overlap=10) as memory:
        await memory.add(
            "u",
            [Message(id="long", content=content), Message(id="other", content="apple pie")],
            request_id="r",
            session_id="s",
        )
        result = await memory.search("u", "apple", options=SearchOptions(mode="lexical"))
        ids = [item.source_id for item in result.evidence]
        assert ids.count("long") > 1
        assert [source.id for source in result.sources] == list(dict.fromkeys(ids))
        assert sorted(source.id for source in result.sources) == ["long", "other"]
        assert next(s for s in result.sources if s.id == "long").content == content


async def test_replay_payload_identity_includes_the_timestamp_offset(tmp_path):
    utc = datetime(2024, 1, 1, 8, 30, tzinfo=UTC)
    paris = utc.astimezone(timezone(timedelta(hours=1)))
    with Memory(tmp_path / "m.db") as memory:

        async def send(when):
            return await memory.add(
                "u",
                [Message(id="m", content="Lunch at noon.", timestamp=when)],
                request_id="r",
                session_id="s",
            )

        assert not (await send(utc)).replayed
        assert (await send(utc)).replayed
        with pytest.raises(IngestConflict):
            await send(paris)
        assert memory.store.stats("u")["sources"] == 1


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


async def test_stale_lexical_connection_cannot_write_into_a_dense_database(tmp_path):
    # Two processes open a fresh path; the dense one records its embedding spec
    # after the lexical one cached "no spec". Chunks written without vectors
    # would be invisible to dense search, so the stale writer must be refused.
    path = tmp_path / "m.db"
    with Memory(path) as stale, Memory(path, embedder=ToyEmbedder()) as dense:
        with pytest.raises(ValueError, match="configuration changed"):
            await add(stale)
        assert dense.store.db.execute("SELECT count(*) FROM evidence_chunks").fetchone()[0] == 0
        await add(dense)
        result = await dense.search("alice", "apple", options=SearchOptions(mode="dense"))
        assert result.evidence


@pytest.mark.parametrize(
    "vector", [[0, 0], [1], [float("nan"), 1], [float("inf"), 1], [1e100, 1], [1e-100, 0]]
)
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        vector_blob(vector, 2)


def test_cosine_is_stable_for_large_finite_vectors(tmp_path):
    with Memory(tmp_path / "large.db") as memory:
        distance = memory.store.db.execute(
            "SELECT vec_distance_cosine(?, ?)",
            (vector_blob([1e38, 1e38], 2), vector_blob([1, 1], 2)),
        ).fetchone()[0]
        assert distance == pytest.approx(0, abs=1e-6)


def test_rrf_uses_ranks_and_retains_both_channels():
    a = Chunk(id="a", source_id="a", index=0, start=0, end=1, text="a")
    b = Chunk(id="b", source_id="b", index=0, start=0, end=1, text="b")
    candidates = fuse(
        {"lexical": [(a, 999), (b, 0.001)], "dense": [(b, 0.9)]},
        weights={"lexical": 1, "dense": 1},
        k=60,
    )
    assert candidates[0].chunk.id == "b"
    assert candidates[0].fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert candidates[0].channels["lexical"].score == 0.001
    assert candidates[0].channels["dense"].rank == 1


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


def test_every_top_level_export_resolves_in_this_install():
    # Holds in both the lean and the legacy install: star-imports resolve
    # every name in __all__, so a legacy name listed without its extra fails.
    package = import_module("mnemotree")
    for name in package.__all__:
        getattr(package, name)


def test_stored_receipt_with_a_newer_field_still_replays():
    receipt = IngestReceipt.model_validate_json(
        '{"namespace":"a","request_id":"r","session_id":"s","source_ids":["x"],'
        '"chunk_count":1,"atom_count":0}'
    )
    assert receipt.source_ids == ("x",)


async def test_lexical_cjk_bigrams_stemming_and_stopwords(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await memory.add(
            "u",
            [
                Message(id="zh", content="我上周搬到了华沙，新公寓在市中心。"),
                Message(id="en", content="Caroline moved to Warsaw after the marathon."),
                Message(id="noise", content="What did you do? Where is it? Who was there?"),
            ],
            request_id="r",
            session_id="s",
        )

        async def ids(query):
            result = await memory.search("u", query, options=SearchOptions(mode="lexical"))
            return [item.source_id for item in result.evidence]

        assert await ids("华沙") == ["zh"]
        assert (await ids("我搬到哪里了？"))[0] == "zh"
        # Porter stemming matches "moving" to "moved"; stopwords do not match noise.
        assert await ids("Where did Caroline end up moving?") == ["en"]
        # A query of only stopwords still searches them.
        assert "noise" in await ids("what is it")


def test_rejects_previous_schema_version(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE evidence_config(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT INTO evidence_config VALUES('schema_version', '1')")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match=r"version '1' \(this build reads '\d+'\); rebuild"):
        Memory(path)
    # Refused before any DDL: the old file keeps exactly its own tables and version.
    db = sqlite3.connect(path)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    version = db.execute("SELECT value FROM evidence_config WHERE key='schema_version'").fetchone()
    db.close()
    assert tables == [("evidence_config",)] and version == ("1",)


async def test_forgotten_chunks_leave_no_lexical_rows_for_new_ingests(tmp_path):
    with Memory(tmp_path / "m.db") as memory:
        await add(memory, "apple orchard", id="old", request="r1")
        assert memory.forget("alice", "old")
        # The next chunk can reuse the forgotten chunk's rowid; a stale FTS row would
        # then make it match the old text.
        await add(memory, "banana grove", id="new", request="r2")
        lexical = SearchOptions(mode="lexical")
        assert not (await memory.search("alice", "apple", options=lexical)).evidence
        found = await memory.search("alice", "banana", options=lexical)
        assert [item.source_id for item in found.evidence] == ["new"]
        db = memory.store.db
        fts = db.execute("SELECT count(*) FROM evidence_fts").fetchone()[0]
        assert fts == db.execute("SELECT count(*) FROM evidence_chunks").fetchone()[0]


async def test_long_queries_are_clipped_for_the_embedder(tmp_path):
    class Recording(ToyEmbedder):
        async def embed_query(self, text):
            self.query = text
            return await super().embed_query(text)

    embedder = Recording()
    with Memory(tmp_path / "memory.db", embedder=embedder) as memory:
        await add(memory)
        query = "apple " + "filler " * 12000 + "orchard question"
        result = await memory.search("alice", query, options=SearchOptions(mode="hybrid"))
    assert result.evidence
    # Clipped by tokens, not characters: well past the character cap, under the token cap.
    assert QUERY_EMBED_CHARS < len(embedder.query) < len(query)
    assert memory.counter.count(embedder.query) <= QUERY_EMBED_TOKENS
    assert embedder.query.startswith("apple ") and embedder.query.endswith("orchard question")


async def test_lexical_search_reads_only_its_namespace(tmp_path):
    from mnemotree.evidence.lexical import scope_token

    # Digits only, so the Porter stemmer cannot merge two namespaces' tokens.
    assert scope_token("alice")[0] == "n" and scope_token("alice")[1:].isdigit()
    assert scope_token("alice") != scope_token("alicee")
    with Memory(tmp_path / "m.db") as memory:
        await add(memory, "apple orchard", namespace="alice", id="a")
        await add(memory, "apple apple apple", namespace="alicee", id="b")
        result = await memory.search("alice", "apple", options=SearchOptions(mode="lexical"))
        assert [item.source_id for item in result.evidence] == ["a"]
        assert memory.forget("alice", "a")
        assert not (
            await memory.search("alice", "apple", options=SearchOptions(mode="lexical"))
        ).evidence


async def test_role_stack_options(tmp_path):
    """One chunk per message, user turns first, and the density-gated deep pool."""
    long_reply = ("apple " * 30 + "\n") * 8  # several chunks, all matching "apple"
    with Memory(tmp_path / "m.db", chunk_size=200, chunk_overlap=20) as memory:
        messages = []
        for i in range(6):
            messages.append(Message(content=f"apple question {i}", role="user", id=f"u{i}"))
            messages.append(Message(content=long_reply, role="assistant", id=f"a{i}"))
        await memory.add("alice", messages, request_id="r", session_id="s")
        assert memory.store.chunk_density("alice") > 2.0

        base = await memory.search(
            "alice", "apple", options=SearchOptions(mode="lexical", limit=100)
        )
        per_source = {}
        for c in base.trace.candidates:
            per_source[c.chunk.source_id] = per_source.get(c.chunk.source_id, 0) + 1
        assert max(per_source.values()) > 1  # long replies take several slots by default

        capped = await memory.search(
            "alice",
            "apple",
            options=SearchOptions(
                mode="lexical", limit=100, chunks_per_source=1, role_first="user"
            ),
        )
        sources = [c.chunk.source_id for c in capped.trace.candidates]
        assert len(sources) == len(set(sources)) == 12
        roles = [item.role for item in capped.evidence]
        assert roles[:6] == ["user"] * 6 and set(roles[6:]) == {"assistant"}

        shallow = SearchOptions(mode="lexical", limit=4, candidate_limit=4)
        deep = SearchOptions(
            mode="lexical",
            limit=4,
            candidate_limit=4,
            deep_candidate_limit=40,
            deep_history_density=2.0,
        )
        assert (await memory.search("alice", "apple", options=shallow)).trace.channel_counts[
            "lexical"
        ] == 4
        assert (await memory.search("alice", "apple", options=deep)).trace.channel_counts[
            "lexical"
        ] > 4
        # A short-turn history stays on the shallow pool.
        await memory.add(
            "bob",
            [Message(content="apple", role="user", id=f"b{i}") for i in range(30)],
            request_id="r",
            session_id="s",
        )
        assert memory.store.chunk_density("bob") == 1.0
        assert (await memory.search("bob", "apple", options=deep)).trace.channel_counts[
            "lexical"
        ] == 4
        with pytest.raises(ValueError, match="deep_candidate_limit"):
            SearchOptions(limit=50, candidate_limit=50, deep_candidate_limit=10)
        # role_first applies only where the assistant carries most of the text.
        gated = SearchOptions(
            mode="lexical",
            limit=100,
            chunks_per_source=1,
            role_first="user",
            role_first_min_assistant_share=0.65,
        )
        assert memory.store.assistant_share("alice") > 0.9
        roles = [
            item.role for item in (await memory.search("alice", "apple", options=gated)).evidence
        ]
        assert roles[:6] == ["user"] * 6
        await memory.add(
            "carol",
            [
                Message(
                    content=f"apple {i} for carol", role=("user", "assistant")[i % 2], id=f"c{i}"
                )
                for i in range(8)
            ],
            request_id="r",
            session_id="s",
        )
        roles = [
            item.role for item in (await memory.search("carol", "apple", options=gated)).evidence
        ]
        assert 0.4 < memory.store.assistant_share("carol") < 0.6
        assert "assistant" in roles[:2]  # a human/human chat keeps fusion order


async def test_capped_chunks_fill_unused_budget(tmp_path):
    with Memory(tmp_path / "m.db", chunk_size=300, chunk_overlap=50) as memory:
        paragraphs = [
            f"Paragraph {i}: the report discusses topic {i} in detail. " * 4 for i in range(8)
        ]
        document = "\n".join(paragraphs)
        await memory.add(
            "dana",
            [Message(content=document, role="user", id="doc")],
            request_id="r",
            session_id="s",
        )
        assert memory.store.chunk_density("dana") > 5
        capped = SearchOptions(
            mode="lexical",
            limit=100,
            chunks_per_source=1,
            token_budget=8000,
            neighbor_messages=1,
            merge_blocks=True,
        )
        result = await memory.search("dana", "report topic 7", options=capped)
        # One message, one item, but the whole document rather than one chunk of it.
        assert len(result.evidence) == 1
        assert "Paragraph 0" in result.context and "Paragraph 7" in result.context
        assert result.trace.packed_tokens > 400
        tight = capped.model_copy(update={"token_budget": 120})
        small = await memory.search("dana", "report topic 7", options=tight)
        assert small.trace.packed_tokens <= 120 and "Paragraph 7" in small.context
        assert small.context.count("Paragraph") < result.context.count("Paragraph")


@pytest.mark.parametrize("encoding", sorted(LINE_SAFE_ENCODINGS))
def test_counter_by_piece_matches_whole_text(encoding):
    counter = TiktokenCounter(encoding)
    parts = ["\n", "\n\n", "\r\n", " ", "  ", "\t", "\u2028", "\xa0", "\u0085", "<|endoftext|>"]
    parts += [".", "/", "[", "]", "-", ":", "'s", "'S", "word", "Word", "12345", "é", "日本"]
    rng = random.Random(0)
    for _ in range(3000):
        text = "".join(rng.choice(parts) for _ in range(rng.randrange(1, 40)))
        assert counter.count(text) == len(counter.encoding.encode(text, disallowed_special=()))


def test_model_query_is_token_aware():
    from mnemotree.evidence.engine import QUERY_EMBED_CHARS, QUERY_EMBED_TOKENS, model_query

    counter = TiktokenCounter("o200k_base")
    latin = (
        "The committee reviewed the quarterly report and noted several open items. " * 160
    ).strip()
    assert len(latin) > QUERY_EMBED_CHARS and counter.count(latin) < QUERY_EMBED_TOKENS
    # Without a counter the character cap applies; with one, the whole query is sent.
    assert "\n...\n" in model_query(latin)
    assert model_query(latin, counter=counter) == latin
    huge = latin * 6
    assert counter.count(huge) > QUERY_EMBED_TOKENS
    cut = model_query(huge, counter=counter)
    assert "\n...\n" in cut and counter.count(cut) <= QUERY_EMBED_TOKENS
    assert cut.startswith(huge[:100]) and cut.endswith(huge[-100:])
    cjk = "华沙是波兰的首都，也是最大的城市。" * 700
    assert counter.count(cjk) > QUERY_EMBED_TOKENS
    assert counter.count(model_query(cjk, counter=counter)) <= QUERY_EMBED_TOKENS


def service_like(**overrides) -> SearchOptions:
    """The AML service defaults (hybrid), for packaging tests with the toy embedder."""
    values = {
        "mode": "hybrid",
        "limit": 100,
        "token_budget": 8000,
        "neighbor_messages": 1,
        "merge_blocks": True,
        "chunks_per_source": 1,
        "role_first": "user",
        "role_first_min_assistant_share": 0.65,
        "candidate_limit": 100,
        "deep_candidate_limit": 300,
        "deep_history_density": 2.0,
    }
    values.update(overrides)
    return SearchOptions(**values)


async def test_packaging_does_not_decide_what_survives(tmp_path):
    """The same information packaged as different transport messages yields the same evidence."""
    steps = [
        f"step {i}: open table t{i} and read column c{i}; observation: value v{i}."
        for i in range(40)
    ]
    final = "final observation: the migration finished with 7 warnings and task 289 is complete."
    trace = "\n".join(steps + [final])
    with Memory(
        tmp_path / "m.db", embedder=ToyEmbedder(), chunk_size=400, chunk_overlap=40
    ) as memory:
        # One trajectory message vs. the same ordered trace as several messages.
        await memory.add(
            "one", [Message(content=trace, role="user", id="t")], request_id="r", session_id="s"
        )
        parts = [
            Message(content="\n".join(steps[i : i + 8]), role="user", id=f"p{i}")
            for i in range(0, 40, 8)
        ]
        parts.append(Message(content=final, role="user", id="pf"))
        await memory.add("many", parts, request_id="r", session_id="s")
        for user in ("one", "many"):
            result = await memory.search(user, "task 289 warnings", options=service_like())
            assert "7 warnings" in result.context, user
            assert "step 0:" in result.context, user  # the record, not just the matching line

        # One document vs. its sections; the needed fact sits in the last section.
        sections = [
            f"Section {i}. " + f"Background material about topic {i}. " * 12 for i in range(6)
        ]
        sections.append("Section 6. Appendix: the reactor's rated output is 412 megawatts.")
        await memory.add(
            "doc",
            [Message(content="\n\n".join(sections), role="user", id="d")],
            request_id="r",
            session_id="s",
        )
        await memory.add(
            "secs",
            [Message(content=text, role="user", id=f"s{i}") for i, text in enumerate(sections)],
            request_id="r",
            session_id="s",
        )
        for user in ("doc", "secs"):
            result = await memory.search(user, "rated output megawatts", options=service_like())
            assert "412 megawatts" in result.context, user

        # A human/human chat as user/assistant pairs vs. all-user with names in content.
        turns = [
            (("Caroline", "Mel")[i % 2], f"chat line {i} about everyday things") for i in range(30)
        ]
        turns[17] = ("Mel", "I went to the LGBTQ support group yesterday and it was powerful.")
        await memory.add(
            "pairs",
            [
                Message(content=t, role=("user", "assistant")[i % 2], id=f"a{i}")
                for i, (_, t) in enumerate(turns)
            ],
            request_id="r",
            session_id="s",
        )
        await memory.add(
            "named",
            [
                Message(content=f"{who}: {t}", role="user", id=f"b{i}")
                for i, (who, t) in enumerate(turns)
            ],
            request_id="r",
            session_id="s",
        )
        assert memory.store.assistant_share("pairs") < 0.65  # a person, so no role reordering
        for user in ("pairs", "named"):
            result = await memory.search(user, "support group", options=service_like())
            assert "LGBTQ support group" in result.context, user
            assert result.evidence[0].source_id in ("a17", "b17"), user

        # A required observation at the end of a long source is not found only on its prefix.
        long = " ".join(f"routine entry {i} with nothing notable." for i in range(300))
        long += " Unexpected: the pressure valve failed at 03:14 and was replaced."
        await memory.add(
            "tail", [Message(content=long, role="user", id="l")], request_id="r", session_id="s"
        )
        result = await memory.search("tail", "pressure valve failed", options=service_like())
        assert "03:14" in result.context
