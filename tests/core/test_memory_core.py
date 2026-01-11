from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core._internal.indexing import BaseQueryExpander, BM25Index
from mnemotree.core._internal.scoring import SignalRanker
from mnemotree.core.memory import (
    MemoryCore,
    ModeDefaultsConfig,
    NerConfig,
    RetrievalConfig,
    ScoringConfig,
)
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.scoring import MemoryScoring
from mnemotree.store.base import BaseMemoryStore


class MockStore(BaseMemoryStore):
    async def store_memory(self, memory):
        pass

    async def get_memory(self, mid):
        return None

    async def delete_memory(self, mid, *, cascade=False):
        return True

    async def list_memories(self, *, include_embeddings=False):
        return []

    async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
        return []

    async def query_memories(self, query, limit=10):
        return []

    async def update_connections(self, memory_id, **kwargs):
        pass

    async def query_by_entities(self, entities, limit=10):
        return []

    async def close(self):
        pass


@pytest.fixture
def mock_store():
    store = MockStore()
    store.store_memory = AsyncMock()
    store.update_connections = AsyncMock()
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mocking behaviors if needed
    return llm


@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    # Mock async embedding call
    embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    return embeddings


@pytest.fixture
def memory_core(mock_store, mock_llm, mock_embeddings):
    return MemoryCore(
        store=mock_store,
        llm=mock_llm,
        embeddings=mock_embeddings,
        ner_config=NerConfig(ner=MagicMock()),
        mode_defaults=ModeDefaultsConfig(mode="pro"),  # Enable summarization and analysis
    )


@pytest.mark.asyncio
async def test_remember_flow(memory_core, mock_store):
    # Setup mocks to avoid actual analysis
    memory_core.analyzer.analyze = AsyncMock(
        return_value=MagicMock(
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
            tags=["test"],
            emotions=["joy"],
            linked_concepts=[],
        )
    )
    memory_core.summarizer.summarize = AsyncMock(return_value="Summary")
    memory_core.ner.extract_entities = AsyncMock(return_value=MagicMock(entities={}, mentions={}))

    memory = await memory_core.remember("Test memory content")

    assert memory.content == "Test memory content"
    assert memory.summary == "Summary"

    # Verify store was called
    mock_store.store_memory.assert_called_once()


@pytest.mark.asyncio
async def test_remember_with_references(memory_core, mock_store):
    # Setup mocks
    memory_core.analyzer.analyze = AsyncMock(
        return_value=MagicMock(
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            tags=[],
            emotions=[],
            linked_concepts=[],
        )
    )
    memory_core.summarizer.summarize = AsyncMock(return_value="Summary")
    memory_core.ner.extract_entities = AsyncMock(return_value=MagicMock(entities={}, mentions={}))

    await memory_core.remember("Test content", references=["prev-id"])

    # Store called
    mock_store.store_memory.assert_called_once()
    # Connect called
    mock_store.update_connections.assert_called_once()


@pytest.mark.asyncio
async def test_connect(memory_core, mock_store):
    await memory_core.connect("mem-1", related_to=["mem-2"])
    mock_store.update_connections.assert_called_with(
        memory_id="mem-1", related_ids=["mem-2"], conflict_ids=None, previous_id=None, next_id=None
    )


@pytest.mark.asyncio
async def test_lite_mode_defaults(mock_store, mock_embeddings):
    class StubKeywordExtractor:
        async def extract(self, text: str):
            return ["alpha", "beta"]

    memory_core = MemoryCore(
        store=mock_store,
        embeddings=mock_embeddings,
        mode_defaults=ModeDefaultsConfig(
            mode="lite",
            keyword_extractor=StubKeywordExtractor(),
        ),
        ner_config=NerConfig(enable_ner=False),
    )

    memory = await memory_core.remember("Lite content")

    assert memory.summary is None
    assert set(memory.tags) == {"alpha", "beta"}
    assert memory.memory_type == MemoryType.SEMANTIC


@pytest.mark.asyncio
async def test_search_uses_bm25_index_when_enabled(mock_store, mock_embeddings):
    memory_core = MemoryCore(
        store=mock_store,
        embeddings=mock_embeddings,
        mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
        ner_config=NerConfig(enable_ner=False),
        retrieval_config=RetrievalConfig(enable_bm25=True),
    )

    m1 = MemoryItem(
        memory_id="m1",
        content="saffron rice",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01 00:00:00.000000+0000",
        embedding=[0.0] * 10,
        last_accessed="2024-01-01 00:00:00.000000+0000",
        tags=[],
    )
    m2 = MemoryItem(
        memory_id="m2",
        content="pasta",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01 00:00:00.000000+0000",
        embedding=[0.0] * 10,
        last_accessed="2024-01-01 00:00:00.000000+0000",
        tags=[],
    )

    # Populate the lexical index cache directly (no need to call remember()).
    memory_core.index_manager.add(m1)
    memory_core.index_manager.add(m2)

    # Ensure store/vector path is not used.
    mock_store.query_memories = AsyncMock(return_value=[])

    results = await memory_core.search("saffron", limit=10)

    assert [m.memory_id for m in results] == ["m1"]
    mock_embeddings.aembed_query.assert_not_called()
    mock_store.query_memories.assert_not_called()


@pytest.mark.asyncio
async def test_recall_scoring_filters_by_relevance(
    mock_store,
    mock_llm,
    mock_embeddings,
):
    mock_embeddings.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    mock_store.get_similar_memories = AsyncMock()

    scoring = MemoryScoring(
        importance_weight=0.0,
        recency_weight=0.0,
        relevance_weight=1.0,
        score_threshold=0.5,
    )
    memory_core = MemoryCore(
        store=mock_store,
        llm=mock_llm,
        embeddings=mock_embeddings,
        scoring_config=ScoringConfig(memory_scoring=scoring),
        ner_config=NerConfig(enable_ner=False),
        mode_defaults=ModeDefaultsConfig(enable_keywords=False),
    )

    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f%z")
    memory_match = MemoryItem(
        memory_id="match",
        content="match",
        memory_type=MemoryType.SEMANTIC,
        importance=0.1,
        timestamp=timestamp,
        embedding=[1.0, 0.0],
    )
    memory_mismatch = MemoryItem(
        memory_id="mismatch",
        content="mismatch",
        memory_type=MemoryType.SEMANTIC,
        importance=0.1,
        timestamp=timestamp,
        embedding=[-1.0, 0.0],
    )
    mock_store.get_similar_memories.return_value = [
        memory_match,
        memory_mismatch,
    ]

    memories = await memory_core.recall("query", scoring=True)

    assert [memory.memory_id for memory in memories] == ["match"]
    mock_store.get_similar_memories.assert_called_once()
    _, kwargs = mock_store.get_similar_memories.call_args
    assert kwargs["query_embedding"] == [1.0, 0.0]


def test_prf_expands_bm25_query_terms():
    index = BM25Index()
    index.add("m1", ["biryani", "saffron", "rice"])
    index.add("m2", ["biryani", "chicken"])

    ranked = index.search(["biryani"], top_k=10)
    expander = BaseQueryExpander(index, top_docs=2, top_terms=2)
    expanded = expander.expand(tokens=["biryani"], top_k_docs=ranked)

    assert expanded[0] == "biryani"
    assert len(expanded) > 1
    assert "saffron" in expanded or "rice" in expanded or "chicken" in expanded


def test_prf_skips_stopword_only_queries():
    index = BM25Index()
    index.add("m1", ["biryani", "saffron", "rice"])
    ranked = [("m1", 1.0)]

    expander = BaseQueryExpander(index, top_docs=1, top_terms=2)
    expanded = expander.expand(tokens=["what", "is"], top_k_docs=ranked)
    assert expanded == ["what", "is"]


def test_rrf_post_rerank_uses_similarity_and_rrf_scores():
    m1 = MemoryItem(
        memory_id="m1",
        content="m1",
        memory_type=MemoryType.SEMANTIC,
        importance=0.1,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
        embedding=[0.0, 1.0],
        tags=["alpha"],
    )
    m2 = MemoryItem(
        memory_id="m2",
        content="m2",
        memory_type=MemoryType.SEMANTIC,
        importance=0.1,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
        embedding=[1.0, 0.0],
        tags=["beta"],
    )

    reranked = SignalRanker().rank(
        [m1, m2],
        query_embedding=[1.0, 0.0],
        extra_signals={
            "query_keywords": [],
            "entity_memory_ids": set(),
            "rrf_scores": {"m1": 1.0, "m2": 0.0},
        },
    )
    assert [m.memory_id for m in reranked] == ["m2", "m1"]
