import json

import httpx
import pytest

from mnemotree.core.models import EmbeddingSpec
from mnemotree.evaluation import ablate
from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.ingest.atoms import AtomDraft, Citation, ExtractorSpec, OpenAIExtractor


class Extractor:
    spec = ExtractorSpec(provider="test", model="fixture", revision="1", prompt_hash="test")

    def __init__(self):
        self.calls = 0
        self.bad_quote = False
        self.inputs = []

    async def extract(self, passages):
        self.calls += 1
        self.inputs.append(passages)
        return [
            AtomDraft(
                text="Maya is a physician in Warsaw.",
                citations=tuple(
                    Citation(passage_id=p.id, quote="invented" if self.bad_quote else p.text)
                    for p in passages
                ),
            )
        ]


class Embedder:
    spec = EmbeddingSpec(provider="test", model="test", revision="1", dimensions=2)

    async def embed_documents(self, texts):
        return [[1, 0] if "physician" in text else [0, 1] for text in texts]

    async def embed_query(self, text):
        return [1, 0]


async def ingest(memory, namespace="u", request="r"):
    return await memory.add(
        namespace,
        [
            Message(id="name", content="I am Maya."),
            Message(id="job", content="I work as a doctor in Warsaw."),
        ],
        request_id=request,
        session_id="s",
    )


async def test_atom_search_resolves_to_original_sources_and_replays(tmp_path):
    extractor = Extractor()
    path = tmp_path / "m.db"
    with Memory(path, extractor=extractor, embedder=Embedder()) as memory:
        receipt = await ingest(memory)
        assert receipt.atom_count == 1
        assert (await ingest(memory)).replayed
        assert extractor.calls == 1
        raw = await memory.search("u", "physician", options=SearchOptions(mode="lexical"))
        assert not raw.evidence
        for mode in ("lexical", "dense", "hybrid"):
            result = await memory.search(
                "u", "physician", options=SearchOptions(mode=mode, representation="atoms")
            )
            assert {item.source_id for item in result.evidence} == {"name", "job"}
            assert all("physician" not in item.text for item in result.evidence)
            assert len(result.trace.atom_hits) == (2 if mode == "hybrid" else 1)
            assert all(
                item.channels[next(iter(item.channels))].rank == 1 for item in result.evidence
            )
        combined = await memory.search(
            "u", "Warsaw", options=SearchOptions(mode="lexical", representation="combined")
        )
        assert len(combined.evidence) == 2
        assert len({item.id for item in combined.evidence}) == 2
        assert any(set(item.channels) == {"lexical", "atom_lexical"} for item in combined.evidence)
        assert combined.trace.packed_tokens <= 8000
    with Memory(path) as reopened:
        assert (
            await reopened.search(
                "u", "physician", options=SearchOptions(mode="lexical", representation="atoms")
            )
        ).evidence
        with pytest.raises(ValueError):
            await ingest(reopened, request="new")


async def test_scope_and_deletion_invalidate_shared_atoms(tmp_path):
    with Memory(tmp_path / "m.db", extractor=Extractor(), embedder=Embedder()) as memory:
        await ingest(memory)
        await ingest(memory, namespace="v")
        assert not (
            await memory.search("other", "physician", options=SearchOptions(representation="atoms"))
        ).evidence
        assert memory.forget("u", "name")
        for mode in ("lexical", "dense"):
            options = SearchOptions(mode=mode, representation="atoms")
            assert not (await memory.search("u", "physician", options=options)).evidence
            assert (await memory.search("v", "physician", options=options)).evidence
        assert memory.source("u", "job") is not None
        assert memory.store.db.execute("SELECT count(*) FROM evidence_atom_vec").fetchone()[0] == 1


async def test_invalid_citations_and_failed_atom_insert_roll_back(tmp_path):
    extractor = Extractor()
    with Memory(tmp_path / "m.db", extractor=extractor, embedder=Embedder()) as memory:
        extractor.bad_quote = True
        with pytest.raises(ValueError, match="exact, unambiguous"):
            await ingest(memory)
        assert memory.store.stats("u")["requests"] == 0
        extractor.bad_quote = False
        memory.store.db.execute(
            "CREATE TRIGGER reject_atoms BEFORE INSERT ON evidence_atoms "
            "BEGIN SELECT RAISE(ABORT, 'test'); END"
        )
        with pytest.raises(ValueError):
            await ingest(memory)
        assert memory.store.stats("u")["sources"] == 0
        assert memory.store.db.execute("SELECT count(*) FROM evidence_vec").fetchone()[0] == 0
        memory.store.db.execute("DROP TRIGGER reject_atoms")
        assert (await ingest(memory)).atom_count == 1


async def test_bounded_extraction_has_no_silent_text_loss(tmp_path):
    extractor = Extractor()
    content = "abcd" * 15
    with Memory(tmp_path / "m.db", extractor=extractor, extraction_batch_chars=17) as memory:
        await memory.add("u", [Message(id="x", content=content)], request_id="r", session_id="s")
        assert "".join(p.text for batch in extractor.inputs for p in batch) == content
        assert all(sum(len(p.text) for p in batch) <= 17 for batch in extractor.inputs)
        atoms = memory.store.db.execute("SELECT payload FROM evidence_atoms").fetchall()
        for row in atoms:
            span = json.loads(row[0])["spans"][0]
            assert content[span["start"] : span["end"]]
    wrong = Extractor()
    wrong.spec = wrong.spec.model_copy(update={"revision": "2"})
    with pytest.raises(ValueError, match="signature mismatch"):
        Memory(tmp_path / "m.db", extractor=wrong, extraction_batch_chars=17)


async def test_explicit_rebuild_and_raw_mode(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path) as memory:
        await ingest(memory)
        with pytest.raises(ValueError, match="extracted corpus"):
            await memory.search(
                "u", "doctor", options=SearchOptions(mode="lexical", representation="atoms")
            )
    with pytest.raises(ValueError, match="extraction rebuild"):
        Memory(path, extractor=Extractor())


async def test_openai_extractor_transport_and_strict_output(monkeypatch, tmp_path):
    actual_client = httpx.AsyncClient
    malformed = False

    def respond(request):
        body = json.loads(request.content)
        passages = json.loads(body["messages"][1]["content"])
        assert body["response_format"] == {"type": "json_object"}
        assert request.headers["authorization"] == "Bearer test-only"
        output = {
            "atoms": [
                {
                    "text": "doctor",
                    "citations": [{"passage_id": passages[0]["id"], "quote": passages[0]["text"]}],
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length" if malformed else "stop",
                        "message": {"content": json.dumps(output)},
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: actual_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    extractor = OpenAIExtractor(model="fixture", revision="1", api_key="test-only")
    with Memory(tmp_path / "m.db", extractor=extractor) as memory:
        assert (await ingest(memory)).atom_count == 1
        malformed = True
        with pytest.raises(ValueError, match="complete normally"):
            await ingest(memory, namespace="v")
        assert memory.store.stats("v")["sources"] == 0
        assert extractor.usage["calls"] == 2
        assert extractor.usage["unreported_calls"] == 2


async def test_same_corpus_ablation_only_extracts_once(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "job", "content": "I am a doctor."}],
            }
        )
    )
    queries.write_text(
        json.dumps(
            {"id": "q", "namespace": "u", "query": "physician", "relevant_source_ids": ["job"]}
        )
    )
    extractor = Extractor()
    with Memory(tmp_path / "m.db", extractor=extractor) as memory:
        report = await ablate(memory, corpus, queries, SearchOptions(mode="lexical"))
        assert extractor.calls == 1
        assert report["summary"]["raw"]["packed_source_recall"] == 0
        assert report["summary"]["atoms"]["packed_source_recall"] == 1
        assert report["summary"]["combined"]["packed_source_recall"] == 1
        assert report["shared_ingestion"]["atoms"] == 1
        assert report["reports"]["atoms"]["manifest"]["extraction"]["extractor"]["revision"] == "1"


async def test_stale_connection_cannot_bypass_projection_configuration(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path) as stale, Memory(path, extractor=Extractor()) as configured:
        with pytest.raises(ValueError, match="configuration changed"):
            await ingest(stale)
        assert (await ingest(configured)).atom_count == 1


async def test_citations_cannot_refer_to_another_batch(tmp_path):
    class Invalid(Extractor):
        async def extract(self, passages):
            return [
                AtomDraft(text="fact", citations=(Citation(passage_id="outside", quote="fact"),))
            ]

    with Memory(tmp_path / "m.db", extractor=Invalid()) as memory:
        with pytest.raises(ValueError, match="outside its extraction batch"):
            await ingest(memory)
        assert memory.store.stats("u")["sources"] == 0
