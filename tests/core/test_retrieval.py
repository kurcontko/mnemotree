import pytest

from mnemotree.core.models import LinkType, MemoryItem, MemoryLink, MemoryType
from mnemotree.core.query import MemoryQuery, MemoryQueryBuilder
from mnemotree.core.retrieval import (
    CAUSAL_LINK_TYPES,
    SEMANTIC_LINK_TYPES,
    TEMPORAL_LINK_TYPES,
    BaseRetriever,
    HybridRetriever,
    VectorEntityRetriever,
    rrf_fuse,
)
from mnemotree.core.scoring import MemoryScoring
from mnemotree.ner.base import BaseNER, NERResult
from mnemotree.rerankers import BaseReranker


def _memory(
    memory_id: str,
    embedding: list[float],
    *,
    tags: list[str] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=f"memory-{memory_id}",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=embedding,
        tags=tags or [],
    )


class DummyEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    async def aembed_query(self, text: str) -> list[float]:
        return self.mapping[text]


class DummyKeywordExtractor:
    def __init__(self, keywords: list[str]):
        self.keywords = keywords
        self.calls: list[str] = []

    async def extract(self, text: str) -> list[str]:
        self.calls.append(text)
        return self.keywords


class DummyNER(BaseNER):
    def __init__(self, entities: dict[str, str]):
        self.entities = entities

    async def extract_entities(self, text: str) -> NERResult:
        mentions = {entity: [text] for entity in self.entities}
        return NERResult(entities=self.entities, mentions=mentions)


class DummyStore:
    def __init__(
        self,
        *,
        vector_memories: list[MemoryItem] | None = None,
        entity_memories: list[MemoryItem] | None = None,
        structured_memories: list[MemoryItem] | None = None,
    ) -> None:
        self.vector_memories = vector_memories or []
        self.entity_memories = entity_memories or []
        self.structured_memories = structured_memories or []
        self.updated_metadata: dict[str, dict[str, object]] = {}
        self.vector_calls: list[tuple[str, list[float], int]] = []
        self.entity_calls: list[list[str]] = []
        self.query_calls: list[MemoryQuery] = []

    async def store_memory(self, memory: MemoryItem) -> None:
        return None

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        return None

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool:
        return False

    async def close(self) -> None:
        return None

    async def get_similar_memories(
        self,
        *,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[MemoryItem]:
        self.vector_calls.append((query, list(query_embedding), top_k))
        return self.vector_memories[:top_k]

    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]:
        if isinstance(entities, dict):
            entities = list(entities.keys())
        self.entity_calls.append(list(entities))
        return self.entity_memories[:limit]

    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        self.query_calls.append(query)
        return self.structured_memories

    async def update_memory_metadata(self, memory_id: str, metadata: dict[str, object]) -> bool:
        self.updated_metadata[memory_id] = metadata
        return True


class DummyIndexManager:
    def __init__(self, ranked: list[tuple[str, float]], memories: list[MemoryItem]):
        self.ranked = ranked
        self.memories = {memory.memory_id: memory for memory in memories}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.ranked[:k]

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        return self.memories.get(memory_id)

    @property
    def doc_count(self) -> int:
        return len(self.memories)


class DummyReranker(BaseReranker):
    def __init__(self, order: dict[str, int]):
        self.order = order
        self.calls: list[tuple[str, list[str], int | None]] = []

    async def rerank(
        self,
        query: str,
        candidates: list[MemoryItem],
        top_k: int | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        self.calls.append((query, [memory.memory_id for memory in candidates], top_k))
        ordered = sorted(
            candidates,
            key=lambda memory: self.order.get(memory.memory_id, 999),
        )
        results = [(memory, 1.0) for memory in ordered]
        if top_k is not None:
            return results[:top_k]
        return results


def test_rrf_fuse_skips_zero_weight_stage():
    m1 = _memory("m1", [1.0, 0.0])
    m2 = _memory("m2", [1.0, 0.0])
    m3 = _memory("m3", [1.0, 0.0])

    fused, scores, stage_scores = rrf_fuse(
        stage_candidates={"vector": [m1, m2], "entity": [m3]},
        weights={"vector": 1.0, "entity": 0.0},
        rrf_k=10,
    )

    assert [memory.memory_id for memory in fused] == ["m1", "m2"]
    assert "entity" not in stage_scores
    assert scores["m1"] > scores["m2"]


@pytest.mark.asyncio
async def test_base_retriever_validation_and_dedupe():
    store = DummyStore()
    retriever = BaseRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=DummyEmbedder({"q": [1.0, 0.0]}),
    )

    with pytest.raises(ValueError):
        await retriever._query_store(MemoryQuery())

    with pytest.raises(ValueError):
        await retriever._query_store("bad-query-type")  # type: ignore[arg-type]

    m1 = _memory("m1", [1.0, 0.0])
    m2 = _memory("m1", [1.0, 0.0])
    deduped = retriever._dedupe_memories([m1, m2])
    assert [memory.memory_id for memory in deduped] == ["m1"]


@pytest.mark.asyncio
async def test_base_retriever_query_store_builder():
    m1 = _memory("m1", [1.0, 0.0])
    store = DummyStore(structured_memories=[m1])
    retriever = BaseRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=DummyEmbedder({"q": [1.0, 0.0]}),
    )

    builder = MemoryQueryBuilder().similar_to(vector=[1.0, 0.0])
    memories, vector = await retriever._query_store(builder)

    assert memories == [m1]
    assert vector == pytest.approx([1.0, 0.0])


@pytest.mark.asyncio
async def test_base_retriever_requires_embedder():
    store = DummyStore()
    retriever = BaseRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=None,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError):
        await retriever._get_embedding("query")


@pytest.mark.asyncio
async def test_vector_entity_recall_with_signals_and_updates():
    query = "find alpha"
    embedder = DummyEmbedder({query: [1.0, 0.0]})
    keyword_extractor = DummyKeywordExtractor(["alpha"])
    ner = DummyNER({"alpha": "ORG"})

    m1 = _memory("m1", [1.0, 0.0])
    m2 = _memory("m2", [1.0, 0.0], tags=["alpha"])
    m3 = _memory("m3", [0.0, 1.0])

    store = DummyStore(vector_memories=[m1, m2], entity_memories=[m2, m3])
    retriever = VectorEntityRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=ner,
        keyword_extractor=keyword_extractor,
        embedder=embedder,
    )

    results = await retriever.recall(query, limit=None, scoring=False, update_access=True)

    assert [memory.memory_id for memory in results] == ["m2", "m1"]
    assert set(store.updated_metadata.keys()) == {"m1", "m2"}


@pytest.mark.asyncio
async def test_vector_entity_recall_structured_query_with_limit():
    m1 = _memory("m1", [1.0, 0.0])
    m2 = _memory("m2", [1.0, 0.0])
    store = DummyStore(structured_memories=[m1, m2])

    retriever = VectorEntityRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=DummyEmbedder({"q": [1.0, 0.0]}),
    )

    query = MemoryQuery(vector=[1.0, 0.0])
    results = await retriever.recall(query, limit=1, scoring=True, update_access=False)

    assert [memory.memory_id for memory in results] == ["m1"]


@pytest.mark.asyncio
async def test_hybrid_fusion_recall_structured_updates_access():
    m1 = _memory("m1", [1.0, 0.0])
    store = DummyStore(structured_memories=[m1])

    retriever = HybridRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=DummyEmbedder({"q": [1.0, 0.0]}),
    )

    query = MemoryQuery(vector=[1.0, 0.0])
    results = await retriever.recall(query, limit=None, scoring=True, update_access=True)

    assert [memory.memory_id for memory in results] == ["m1"]
    assert set(store.updated_metadata.keys()) == {"m1"}


@pytest.mark.asyncio
async def test_hybrid_fusion_recall_reranks_top_candidates():
    query = "rerank alpha"
    embedder = DummyEmbedder({query: [1.0, 0.0]})
    keyword_extractor = DummyKeywordExtractor(["alpha"])
    ner = DummyNER({"alpha": "ORG"})

    m1 = _memory("m1", [1.0, 0.0])
    m2 = _memory("m2", [1.0, 0.0], tags=["alpha"])
    m3 = _memory("m3", [1.0, 0.0])

    store = DummyStore(vector_memories=[m1, m2, m3], entity_memories=[m2])
    index_manager = DummyIndexManager(
        ranked=[("m3", 1.0), ("m1", 0.5)],
        memories=[m1, m2, m3],
    )
    reranker = DummyReranker({"m2": 0, "m1": 1, "m3": 2})

    retriever = HybridRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=ner,
        keyword_extractor=keyword_extractor,
        embedder=embedder,
        index_manager=index_manager,
        enable_rrf_signal_rerank=True,
        reranker=reranker,
        rerank_candidates=2,
    )

    results = await retriever.recall(query, limit=3, scoring=False, update_access=True)

    assert [memory.memory_id for memory in results][:2] == ["m2", "m1"]
    assert reranker.calls[0][2] is None
    assert set(store.updated_metadata.keys()) == {"m1", "m2", "m3"}


# ---------------------------------------------------------------------------
# MAGMA four-graph constants
# ---------------------------------------------------------------------------


class TestGraphDimensionConstants:
    def test_semantic_link_types(self) -> None:
        assert LinkType.SUPPORTS in SEMANTIC_LINK_TYPES
        assert LinkType.CONTRADICTS in SEMANTIC_LINK_TYPES
        assert LinkType.ELABORATES in SEMANTIC_LINK_TYPES
        assert LinkType.SIMILAR_TO in SEMANTIC_LINK_TYPES
        assert LinkType.GENERALIZES in SEMANTIC_LINK_TYPES
        assert LinkType.SUPERSEDES in SEMANTIC_LINK_TYPES

    def test_temporal_link_types(self) -> None:
        assert LinkType.FOLLOWS in TEMPORAL_LINK_TYPES
        assert LinkType.SEQUENCE in TEMPORAL_LINK_TYPES
        assert len(TEMPORAL_LINK_TYPES) == 2

    def test_causal_link_types(self) -> None:
        assert LinkType.CAUSES in CAUSAL_LINK_TYPES
        assert LinkType.DERIVES_FROM in CAUSAL_LINK_TYPES
        assert len(CAUSAL_LINK_TYPES) == 2

    def test_no_overlap_temporal_semantic(self) -> None:
        assert not (TEMPORAL_LINK_TYPES & SEMANTIC_LINK_TYPES)


# ---------------------------------------------------------------------------
# Graph-aware DummyStore for MAGMA traversal tests
# ---------------------------------------------------------------------------


class GraphAwareDummyStore(DummyStore):
    """Extends DummyStore with SupportsKnowledgeGraph-compatible methods."""

    def __init__(
        self,
        *,
        vector_memories: list[MemoryItem] | None = None,
        entity_memories: list[MemoryItem] | None = None,
        graph_neighbors: list[tuple[MemoryItem, int, list[MemoryLink]]] | None = None,
    ) -> None:
        super().__init__(
            vector_memories=vector_memories,
            entity_memories=entity_memories,
        )
        self.graph_neighbors = graph_neighbors or []
        self.traverse_calls: list[dict] = []

    async def traverse_graph(
        self,
        start_id: str,
        *,
        max_depth: int = 3,
        link_types: list[LinkType] | None = None,
        strategy: str = "bfs",
    ) -> list[tuple[MemoryItem, int, list[MemoryLink]]]:
        self.traverse_calls.append({
            "start_id": start_id,
            "max_depth": max_depth,
            "link_types": link_types,
        })
        return self.graph_neighbors

    async def create_link(self, *a, **kw) -> MemoryLink:
        raise NotImplementedError

    async def get_links(self, *a, **kw) -> list[MemoryLink]:
        return []

    async def find_path(self, *a, **kw):
        return None

    async def suggest_links(self, *a, **kw):
        return []

    async def update_link_strength(self, *a, **kw) -> bool:
        return True

    async def delete_link(self, *a, **kw) -> bool:
        return True

    async def get_backlinks(self, *a, **kw) -> list[MemoryItem]:
        return []

    async def get_neighborhood_links(self, *a, **kw) -> list[MemoryLink]:
        return []

    async def traverse_typed_path(self, *a, **kw) -> list:
        return []


@pytest.mark.asyncio
async def test_hybrid_retriever_collects_graph_candidates() -> None:
    """HybridRetriever collects graph candidates when graph_weight > 0."""
    query = "test query"
    embedder = DummyEmbedder({query: [1.0, 0.0]})
    m1 = _memory("m1", [1.0, 0.0])
    m2 = _memory("m2", [0.9, 0.1])
    m_graph = _memory("g1", [0.8, 0.2])

    link = MemoryLink(
        source_id="m1",
        target_id="g1",
        link_type=LinkType.ELABORATES,
    )
    store = GraphAwareDummyStore(
        vector_memories=[m1, m2],
        graph_neighbors=[(m_graph, 1, [link])],
    )

    retriever = HybridRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=embedder,
        graph_weight=0.3,
        vector_weight=0.5,
        bm25_weight=0.2,
    )

    results = await retriever.recall(query, limit=5, scoring=False, update_access=False)
    # Graph candidate should be included in results
    result_ids = {m.memory_id for m in results}
    assert "g1" in result_ids
    # Store's traverse_graph should have been called
    assert len(store.traverse_calls) > 0


@pytest.mark.asyncio
async def test_hybrid_retriever_skips_graph_when_weight_zero() -> None:
    """HybridRetriever skips graph collection when graph_weight=0."""
    query = "test query"
    embedder = DummyEmbedder({query: [1.0, 0.0]})
    m1 = _memory("m1", [1.0, 0.0])

    store = GraphAwareDummyStore(vector_memories=[m1])

    retriever = HybridRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=embedder,
        graph_weight=0.0,
    )

    await retriever.recall(query, limit=5, scoring=False, update_access=False)
    assert len(store.traverse_calls) == 0


@pytest.mark.asyncio
async def test_graph_candidates_with_none_entities():
    """Memories with entities=None should not crash graph candidate collection."""
    m1 = _memory("m1", [1.0, 0.0])
    m1.entities = None  # type: ignore[assignment]
    m2 = _memory("m2", [0.9, 0.1])
    m2.entities = {"org": "ACME"}

    store = DummyStore(vector_memories=[m1, m2])
    embedder = DummyEmbedder({"q": [1.0, 0.0]})
    retriever = HybridRetriever(
        store=store,
        scoring_system=MemoryScoring(),
        ner=None,
        keyword_extractor=None,
        embedder=embedder,
        graph_weight=0.3,
    )
    # Should not raise AttributeError: 'NoneType' object has no attribute 'keys'
    results = await retriever.recall(
        MemoryQuery(vector=[1.0, 0.0]), limit=5, scoring=False, update_access=False
    )
    assert isinstance(results, list)
