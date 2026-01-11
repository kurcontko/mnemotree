import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.retrieval import (
    BaseReranker,
    FusionStrategy,
    HybridRetriever,
    RetrievalStage,
)


def _memory(memory_id: str, content: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=[0.1, 0.2],
    )


class ScoreReranker(BaseReranker):
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    async def rerank(
        self,
        query: str,
        candidates: list[MemoryItem],
        top_k: int | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        ordered = sorted(
            candidates,
            key=lambda mem: self.scores.get(mem.memory_id, 0.0),
            reverse=True,
        )
        results = [(mem, self.scores[mem.memory_id]) for mem in ordered]
        if top_k is not None:
            return results[:top_k]
        return results


@pytest.mark.asyncio
async def test_weighted_sum_fusion_rewards_multi_signal_candidates():
    retriever = HybridRetriever(
        vector_weight=0.7,
        entity_weight=0.3,
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
    )
    m1 = _memory("m1", "vector")
    m2 = _memory("m2", "entity")

    results = await retriever.retrieve(
        query="query",
        vector_candidates=[(m1, 0.9), (m2, 0.8)],
        entity_candidates=[(m2, 0.95)],
        graph_candidates=None,
        top_k=2,
        apply_reranking=False,
    )

    assert [r.memory.memory_id for r in results] == ["m2", "m1"]
    assert set(results[0].scores.keys()) == {RetrievalStage.VECTOR.value, RetrievalStage.ENTITY.value}


@pytest.mark.asyncio
async def test_rrf_fusion_tracks_stage_provenance():
    retriever = HybridRetriever(
        vector_weight=0.5,
        entity_weight=0.3,
        graph_weight=0.2,
        fusion_strategy=FusionStrategy.RRF,
    )
    m1 = _memory("m1", "vector-first")
    m2 = _memory("m2", "entity-first")
    m3 = _memory("m3", "graph-only")

    results = await retriever.retrieve(
        query="query",
        vector_candidates=[(m1, 0.9), (m2, 0.7)],
        entity_candidates=[(m2, 0.95), (m1, 0.5)],
        graph_candidates=[(m3, 0.8)],
        top_k=3,
        apply_reranking=False,
    )

    assert {results[0].memory.memory_id, results[1].memory.memory_id} == {"m1", "m2"}
    assert results[2].memory.memory_id == "m3"

    result_map = {r.memory.memory_id: r for r in results}
    assert RetrievalStage.VECTOR in result_map["m1"].retrieval_stages
    assert RetrievalStage.ENTITY in result_map["m2"].retrieval_stages
    assert RetrievalStage.GRAPH in result_map["m3"].retrieval_stages


@pytest.mark.asyncio
async def test_reranker_adjusts_final_ordering_when_enabled():
    retriever = HybridRetriever(
        vector_weight=1.0,
        entity_weight=0.0,
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
        reranker=ScoreReranker({"m1": 0.1, "m2": 0.9}),
    )
    m1 = _memory("m1", "higher vector score")
    m2 = _memory("m2", "reranker favorite")

    results = await retriever.retrieve(
        query="query",
        vector_candidates=[(m1, 0.95), (m2, 0.8)],
        entity_candidates=[],
        graph_candidates=None,
        top_k=2,
        apply_reranking=True,
    )

    assert [r.memory.memory_id for r in results] == ["m2", "m1"]
    assert RetrievalStage.RERANK in results[0].retrieval_stages
