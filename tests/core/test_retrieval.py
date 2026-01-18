import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.query import MemoryQuery, MemoryQueryBuilder
from mnemotree.core.retrieval import (
    BaseRetriever,
    HybridFusionRetriever,
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
    assert vector == [1.0, 0.0]


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

    retriever = HybridFusionRetriever(
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

    retriever = HybridFusionRetriever(
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
