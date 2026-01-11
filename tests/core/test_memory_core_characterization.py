from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core.memory import MemoryCore, ModeDefaultsConfig, NerConfig, RetrievalConfig
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.store.base import BaseMemoryStore

# --- Mocks ---


class MockStore(BaseMemoryStore):
    def __init__(self):
        self.stored_memories = {}

    async def store_memory(self, memory):
        self.stored_memories[memory.memory_id] = memory

    async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
        # Return empty by default, tests will override or populate
        return []

    async def query_memories(self, query):
        return []

    async def update_connections(self, memory_id, **kwargs):
        pass

    async def query_by_entities(self, entities):
        return []

    async def update_memory_metadata(self, memory_id, metadata):
        pass

    async def delete_memory(self, mid, cascade=False):
        return True

    async def get_memory(self, mid):
        return self.stored_memories.get(mid)

    async def close(self):
        pass


@pytest.fixture
def char_store():
    return MockStore()


@pytest.fixture
def char_embeddings():
    embeds = MagicMock()

    # Return deterministic embedding based on text length to allow differentiation
    async def _embed(text):
        val = len(text) % 10 / 10.0
        return [val] * 10  # 10-dim embedding

    embeds.aembed_query = AsyncMock(side_effect=_embed)
    return embeds


@pytest.fixture
def char_ner():
    ner = MagicMock()

    async def _extract(text):
        if "entity" in text:
            return MagicMock(entities={"entity": "TYPE"}, mentions={"entity": [(0, 6)]})
        return MagicMock(entities={}, mentions={})

    ner.extract_entities = AsyncMock(side_effect=_extract)
    return ner


# --- Characterization Tests ---


@pytest.mark.asyncio
async def test_char_remember_signature(char_store, char_embeddings, char_ner):
    """
    Freeze the exact output structure of remember() including defaults and tag merging.
    """
    mock_kw = MagicMock()
    mock_kw.extract = AsyncMock(return_value=["kw1", "kw2"])

    core = MemoryCore(
        store=char_store,
        embeddings=char_embeddings,
        mode_defaults=ModeDefaultsConfig(
            mode="lite",  # Deterministic, no LLM
            enable_keywords=True,
            keyword_extractor=mock_kw,
        ),
        ner_config=NerConfig(ner=char_ner),
    )

    # Mock keyword extractor to be deterministic
    core.keyword_extractor = mock_kw
    if core.enrichment:
        core.enrichment.keyword_extractor = mock_kw

    memory = await core.remember(
        content="This contains an entity.",
        tags=["manual1"],
        context={"origin": "test"},
        importance=0.8,
    )

    # Assert specific fields to freeze behavior
    assert memory.content == "This contains an entity."
    assert memory.importance == 0.8
    assert memory.memory_type == MemoryType.SEMANTIC
    # Tags should merge manual + keywords
    assert set(memory.tags) == {"manual1", "kw1", "kw2"}
    # Context preserved
    assert memory.context == {"origin": "test"}
    # NER ran using the mock
    assert "entity" in memory.entities
    # Embedding len
    assert len(memory.embedding) == 10

    # Verify exact ID length format (uuid is 36 chars)
    assert len(memory.memory_id) == 36


@pytest.mark.asyncio
async def test_char_recall_legacy_ordering(char_store, char_embeddings):
    """
    Freeze basic (VectorEntity) retrieval ordering: Vector score is primary.
    """
    core = MemoryCore(
        store=char_store,
        embeddings=char_embeddings,
        ner_config=NerConfig(enable_ner=False),
        mode_defaults=ModeDefaultsConfig(
            enable_keywords=False,  # Ensure signal reranking doesn't disturb vector order on ties
        ),
    )

    # Setup store to return specific memories with known embeddings
    # Query: "query" (len 5 -> 0.5)
    # Match 1: "match" (len 5 -> 0.5) -> High sim
    # Match 2: "no" (len 2 -> 0.2) -> Low sim

    m1 = MemoryItem(
        memory_id="m1",
        content="match",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01 00:00:00.000000+0000",
        embedding=[0.5] * 10,
        last_accessed="2024-01-01 00:00:00.000000+0000",
    )
    m2 = MemoryItem(
        memory_id="m2",
        content="no",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01 00:00:00.000000+0000",
        embedding=[0.2] * 10,
        last_accessed="2024-01-01 00:00:00.000000+0000",
    )

    char_store.get_similar_memories = AsyncMock(return_value=[m1, m2])

    results = await core.recall("query", scoring=False)

    assert len(results) == 2
    assert results[0].memory_id == "m1"
    assert results[1].memory_id == "m2"


@pytest.mark.asyncio
async def test_char_recall_rrf_ordering(char_store, char_embeddings):
    """
    Freeze Hybrid (RRF) retrieval ordering: Vector + BM25 + Entity.
    """
    core = MemoryCore(
        store=char_store,
        embeddings=char_embeddings,
        ner_config=NerConfig(enable_ner=False),  # Simplify for this test
        mode_defaults=ModeDefaultsConfig(enable_keywords=False),
        retrieval_config=RetrievalConfig(
            retrieval_mode="hybrid",
            enable_bm25=True,
            rrf_k=60,
        ),
    )

    # Setup:
    # m1: High vector rank, Low BM25
    # m2: Low vector rank, High BM25

    m1 = MemoryItem(
        memory_id="m1",
        content="vec_fav",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01 00:00:00.000000+0000",
        embedding=[0.5] * 10,
        last_accessed="2024-01-01 00:00:00.000000+0000",
    )
    m2 = MemoryItem(
        memory_id="m2",
        content="bm25_fav",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01 00:00:00.000000+0000",
        embedding=[0.2] * 10,
        last_accessed="2024-01-01 00:00:00.000000+0000",
    )

    # Vector returns [m1, m2]
    char_store.get_similar_memories = AsyncMock(return_value=[m1, m2])

    # BM25 returns [m2, m1] (via cache/index)
    # We populate the index manually accessing index_manager
    core.index_manager._cache["m1"] = m1
    core.index_manager._cache["m2"] = m2

    # Mock _bm25_index.search to favor m2
    # m2 rank 1, m1 rank 2
    core.index_manager._index = MagicMock()
    core.index_manager._index.search.return_value = [("m2", 10.0), ("m1", 5.0)]

    results = await core.recall("query", scoring=False)

    # RRF should likely favor m2 because BM25 rank 1 vs 2 is a big diff?
    # Or m1 because Vector weight is 0.6 vs BM25 0.3?
    # Vector: m1(1st), m2(2nd) -> m1 score += 0.6/(60+1), m2 += 0.6/(60+2)
    # BM25: m2(1st), m1(2nd) -> m2 score += 0.3/(60+1), m1 += 0.3/(60+2)
    # m1 = 0.6/61 + 0.3/62 ~= 0.0098 + 0.0048 = 0.0146
    # m2 = 0.6/62 + 0.3/61 ~= 0.0096 + 0.0049 = 0.0145
    # So m1 should win slightly due to higher vector weight (0.6 vs 0.3).

    assert results[0].memory_id == "m1"
    assert results[1].memory_id == "m2"


@pytest.mark.asyncio
async def test_char_bm25_prf_behavior(char_store, char_embeddings):
    """
    Freeze BM25 PRF expansion behavior.
    """
    core = MemoryCore(
        store=char_store,
        embeddings=char_embeddings,
        ner_config=NerConfig(enable_ner=False),
        mode_defaults=ModeDefaultsConfig(enable_keywords=False),
        retrieval_config=RetrievalConfig(
            retrieval_mode="hybrid",
            enable_bm25=True,
            enable_prf=True,
            prf_docs=1,
            prf_terms=1,
        ),
    )

    # Add memory for PRF: content has "saffron", query is "rice".
    # Access index_manager
    core.index_manager._index = MagicMock()
    core.index_manager._index.term_freqs = {"m1": {"rice": 1, "saffron": 1}}
    core.index_manager._index.doc_freq = {"rice": 1, "saffron": 1}
    core.index_manager._index.doc_len = {"m1": 2}
    core.index_manager._index.total_len = 2

    # Search returns m1
    core.index_manager._index.search.return_value = [("m1", 1.5)]

    # Expand
    expander = core.index_manager._expander
    assert expander is not None
    # Fix: update expander's index reference too
    expander.index = core.index_manager._index

    expanded = expander.expand(
        tokens=["rice"],
        top_k_docs=[("m1", 1.5)],
    )

    # Should include saffron
    assert "saffron" in expanded
    assert "rice" in expanded
