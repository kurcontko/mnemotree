"""One namespace's writes must leave every other namespace's retrieval unchanged.

Eligibility alone is not enough: BM25 statistics computed across namespaces let one
user's data reorder another user's results without either seeing the other's records.
"""

import asyncio
import hashlib
import re

import pytest

from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.evidence.atom_store import ATOM_FTS
from mnemotree.evidence.atoms import AtomDraft, Citation, ExtractorSpec
from mnemotree.evidence.lexical import fts_table
from mnemotree.evidence.models import EmbeddingSpec

ALICE = [
    ("user", "I keep an apple in my bag for the commute."),
    ("assistant", "A good snack for the train."),
    ("user", "My banana fruit fruit smoothie recipe needs oats."),
    ("assistant", "Oats make it thicker."),
    ("user", "The cherry fruit tree in the yard bloomed in April."),
    ("assistant", "April is early for that."),
    ("user", "I painted the fence green last weekend."),
    ("user", "The neighbours asked about the fence paint."),
]
BOB = [f"apple apple orchard report {index} and an apple pie" for index in range(40)] + [
    f"cherry cherry harvest log {index} from the cold store" for index in range(40)
]
QUERIES = [
    ("apple banana cherry", None),
    ("what fruit did I talk about", None),
    ("fence paint", ["A. green", "B. apple"]),
]


class HashEmbedder:
    """Deterministic bag of words: no network, and dense ranks follow the text."""

    spec = EmbeddingSpec(provider="test", model="hash", revision="1", dimensions=32)

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.spec.dimensions
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % len(vector)] += 1.0 if digest[1] & 1 else -1.0
        return vector if any(vector) else [1.0] + [0.0] * (len(vector) - 1)

    async def embed_documents(self, texts):
        return [self.vector(text) for text in texts]

    async def embed_query(self, text):
        return self.vector(text)


class OneAtomPerPassage:
    spec = ExtractorSpec(provider="test", model="fixture", revision="1", prompt_hash="test")

    async def extract(self, passages):
        return [
            AtomDraft(text=p.text, citations=(Citation(passage_id=p.id, quote=p.text),))
            for p in passages
        ]


def aml_preset(memory: Memory) -> SearchOptions:
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import aml_options

    return aml_options(memory)


def snapshot(result) -> dict:
    """Everything a caller can observe except wall-clock timings."""
    dump = result.model_dump(mode="json")
    dump["trace"].pop("timings_ms")
    return dump


async def write_alice(memory: Memory, namespace: str = "alice") -> None:
    await memory.add(
        namespace,
        [
            Message(id=f"alice-{index}", role=role, content=content)
            for index, (role, content) in enumerate(ALICE)
        ],
        request_id="alice-1",
        session_id="s1",
    )


async def write_bob(memory: Memory, namespace: str = "bob") -> list[str]:
    ids = [f"bob-{index}" for index in range(len(BOB))]
    await memory.add(
        namespace,
        [
            Message(id=source_id, role="user", content=text)
            for source_id, text in zip(ids, BOB, strict=True)
        ],
        request_id="bob-1",
        # The same session ID as Alice's: sessions are organised within a namespace.
        session_id="s1",
    )
    return ids


@pytest.mark.parametrize("hybrid", [False, True], ids=["lexical", "hybrid"])
async def test_additions_updates_and_deletions_leave_other_namespaces_unchanged(tmp_path, hybrid):
    with Memory(tmp_path / "m.db", embedder=HashEmbedder() if hybrid else None) as memory:
        options = aml_preset(memory)

        async def alice() -> list[dict]:
            return [
                snapshot(
                    await memory.search("alice", query, options=options, query_options=answers)
                )
                for query, answers in QUERIES
            ]

        await write_alice(memory)
        before = await alice()
        assert all(result["evidence"] for result in before)

        ids = await write_bob(memory)
        assert await alice() == before, "Bob's additions changed Alice's retrieval"

        memory.forget("bob", ids[0])
        await memory.add(
            "bob",
            [Message(id="bob-updated", role="user", content="apple banana cherry revised")],
            request_id="bob-2",
            session_id="s2",
        )
        assert await alice() == before, "Bob's update changed Alice's retrieval"

        for source_id in ids[1:]:
            assert memory.forget("bob", source_id)
        assert await alice() == before, "Bob's deletions changed Alice's retrieval"

        assert memory.forget("bob", "bob-updated")
        assert await alice() == before, "emptying Bob's namespace changed Alice's retrieval"
        lexical = SearchOptions(mode="lexical")
        assert not (await memory.search("bob", "apple", options=lexical)).evidence


async def test_arrival_order_between_namespaces_does_not_change_retrieval(tmp_path):
    query, answers = QUERIES[0]

    async def alice_after(writes, name: str) -> dict:
        with Memory(tmp_path / f"{name}.db", embedder=HashEmbedder()) as memory:
            await writes(memory)
            return snapshot(
                await memory.search(
                    "alice", query, options=aml_preset(memory), query_options=answers
                )
            )

    alone = await alice_after(write_alice, "alone")

    async def bob_first(memory: Memory) -> None:
        await write_bob(memory)
        await write_alice(memory)

    async def concurrent(memory: Memory) -> None:
        await asyncio.gather(write_bob(memory), write_alice(memory))

    assert await alice_after(bob_first, "bob-first") == alone
    assert await alice_after(concurrent, "concurrent") == alone


async def test_another_namespaces_atoms_cannot_change_atom_retrieval(tmp_path):
    options = SearchOptions(mode="lexical", representation="atoms")
    with Memory(
        tmp_path / "m.db", extractor=OneAtomPerPassage(), extraction_batch_chars=1000
    ) as memory:

        async def alice() -> dict:
            return snapshot(await memory.search("alice", "apple cherry", options=options))

        await write_alice(memory)
        before = await alice()
        assert before["evidence"]
        ids = await write_bob(memory)
        assert await alice() == before, "Bob's atoms changed Alice's retrieval"
        for source_id in ids:
            memory.forget("bob", source_id)
        assert await alice() == before, "deleting Bob's atoms changed Alice's retrieval"
        assert not (await memory.search("bob", "apple", options=options)).evidence


async def test_forgetting_removes_atoms_from_the_namespace_index(tmp_path):
    options = SearchOptions(mode="lexical", representation="atoms")
    with Memory(tmp_path / "m.db", extractor=OneAtomPerPassage()) as memory:
        await write_alice(memory)
        db = memory.store.db
        table = fts_table(ATOM_FTS, "alice")
        assert memory.forget("alice", "alice-0")
        assert not (await memory.search("alice", "commute", options=options)).evidence
        indexed = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert indexed == db.execute("SELECT count(*) FROM evidence_atoms").fetchone()[0]


async def test_a_namespace_without_records_searches_empty(tmp_path):
    with Memory(tmp_path / "m.db", extractor=OneAtomPerPassage()) as memory:
        for representation in ("raw", "atoms"):
            options = SearchOptions(mode="lexical", representation=representation)
            assert not (await memory.search("nobody", "apple", options=options)).evidence


async def test_namespace_text_never_reaches_sql_as_a_name(tmp_path):
    hostile = "../x'; DROP TABLE evidence_chunks; --"
    options = SearchOptions(mode="lexical")
    with Memory(tmp_path / "m.db") as memory:
        await write_alice(memory, hostile)
        await write_alice(memory)
        found = await memory.search(hostile, "apple", options=options)
        assert [item.source_id for item in found.evidence] == ["alice-0"]
        assert (await memory.search("alice", "apple", options=options)).evidence


async def test_a_failed_first_write_leaves_the_namespace_usable(tmp_path, monkeypatch):
    def fail(*_args):
        raise RuntimeError("stop after the chunks are indexed")

    options = SearchOptions(mode="lexical")
    with Memory(tmp_path / "m.db") as memory:
        # The transaction that created the namespace's index rolls back with it.
        with monkeypatch.context() as patch:
            patch.setattr(memory.store.atoms, "insert", fail)
            with pytest.raises(RuntimeError):
                await write_alice(memory)
        assert not (await memory.search("alice", "apple", options=options)).evidence
        await write_alice(memory)
        assert (await memory.search("alice", "apple", options=options)).evidence
