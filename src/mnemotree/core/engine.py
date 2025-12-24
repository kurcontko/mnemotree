from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models_v2 import MemoryEdge, MemoryEnvelope, MemoryHit, MemoryQuery, MemoryRecord
from .protocols import DocStore, Embedder, Enricher, GraphStore, VectorIndex


class MemoryEngine:
    def __init__(
        self,
        *,
        docstore: DocStore,
        vector: Optional[VectorIndex] = None,
        graph: Optional[GraphStore] = None,
        embedder: Optional[Embedder] = None,
        enrichers: Optional[List[Enricher]] = None,
    ):
        self.docstore = docstore
        self.vector = vector
        self.graph = graph
        self.embedder = embedder
        self.enrichers = enrichers or []

    async def remember(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> MemoryEnvelope:
        env = MemoryEnvelope(
            record=MemoryRecord(
                text=text,
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
                thread_id=thread_id,
            )
        )

        if self.embedder and self.vector:
            env.embedding = await self.embedder.embed(text)

        for enricher in self.enrichers:
            env = await enricher.enrich(env)

        await self.docstore.upsert(env)

        if self.vector and env.embedding is not None:
            payload = {
                "user_id": env.record.user_id,
                "session_id": env.record.session_id,
                "thread_id": env.record.thread_id,
                "tags": env.ann.tags,
                "importance": env.ann.importance,
            }
            await self.vector.upsert(env.record.id, env.embedding, payload)

        if self.graph and env.edges:
            await self.graph.upsert_edges(env.edges)

        return env

    async def recall(self, query: MemoryQuery) -> List[MemoryHit]:
        if (
            query.embedding is None
            and query.text
            and self.embedder
            and self.vector
        ):
            query.embedding = await self.embedder.embed(query.text)

        hits: List[MemoryHit] = []

        if self.vector and query.embedding is not None:
            ids_scores = await self.vector.query(
                query.embedding, query.filters, query.limit
            )
            for memory_id, score in ids_scores:
                env = await self.docstore.get(memory_id)
                if env is None:
                    continue
                hits.append(
                    MemoryHit(memory=env, score=score, reasons={"vector": score})
                )

        if self.graph and query.hops > 0 and hits:
            seed_ids = [h.memory.record.id for h in hits]
            neighbor_ids = await self.graph.neighbors(
                seed_ids, query.hops, query.edge_kinds
            )
            for memory_id in neighbor_ids:
                env = await self.docstore.get(memory_id)
                if env is None:
                    continue
                hits.append(MemoryHit(memory=env, score=0.0, reasons={"graph": True}))

        return hits[: query.limit]

    async def connect(self, edges: List[MemoryEdge]) -> None:
        if not self.graph:
            raise RuntimeError("GraphStore not configured")
        await self.graph.upsert_edges(edges)

