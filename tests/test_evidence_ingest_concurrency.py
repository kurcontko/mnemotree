import asyncio

import pytest

from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.evidence.atoms import AtomDraft, Citation, ExtractorSpec
from mnemotree.evidence.models import EmbeddingSpec


class CharCounter:
    signature = "chars"

    def count(self, text: str) -> int:
        return len(text)


class ProviderDown(RuntimeError):
    pass


class Embedder:
    spec = EmbeddingSpec(provider="test", model="test", revision="1", dimensions=2)

    def __init__(self, started=None, fail=False):
        self.started, self.fail = started, fail

    async def embed_documents(self, texts):
        if self.started is not None:
            # Wait until extraction is in flight too; sequential calls never get here.
            await asyncio.wait_for(self.started.wait(), timeout=5)
        if self.fail:
            try:
                raise OSError("connection reset")
            except OSError as exc:
                raise ProviderDown("embedding failed") from exc
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        return [1.0, 0.0]


class Extractor:
    spec = ExtractorSpec(provider="test", model="fixture", revision="1", prompt_hash="test")

    def __init__(self, started=None, fail=False):
        self.started, self.fail = started, fail

    async def extract(self, passages):
        if self.started is not None:
            self.started.set()
        await asyncio.sleep(0)
        if self.fail:
            raise ValueError("extractor rejected the batch")
        return [
            AtomDraft(text=p.text, citations=(Citation(passage_id=p.id, quote=p.text),))
            for p in passages
        ]


def memory(path, embedder, extractor):
    return Memory(path, embedder=embedder, extractor=extractor, token_counter=CharCounter())


async def add(memory):
    return await memory.add(
        "u", [Message(id="m", content="I live in Lisbon.")], request_id="r", session_id="s"
    )


async def test_chunk_embedding_and_extraction_run_concurrently(tmp_path):
    started = asyncio.Event()
    with memory(tmp_path / "m.db", Embedder(started), Extractor(started)) as m:
        receipt = await add(m)
        assert (receipt.chunk_count, receipt.atom_count) == (1, 1)


@pytest.mark.parametrize("failing", ["embedder", "extractor"])
async def test_a_provider_failure_surfaces_unwrapped_and_writes_nothing(tmp_path, failing):
    embedder = Embedder(fail=failing == "embedder")
    extractor = Extractor(fail=failing == "extractor")
    with memory(tmp_path / "m.db", embedder, extractor) as m:
        if failing == "embedder":
            with pytest.raises(ProviderDown) as raised:
                await add(m)
            # The provider's own cause survives, for the service's failure logging.
            assert isinstance(raised.value.__cause__, OSError)
        else:
            with pytest.raises(ValueError, match="extractor rejected"):
                await add(m)
        found = await m.search("u", "Lisbon", options=SearchOptions(mode="lexical"))
        assert not found.evidence
        assert m.source("u", "m") is None
