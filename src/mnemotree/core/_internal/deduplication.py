from __future__ import annotations

import logging

from ..models import MemoryItem
from ..retrieval import Retriever
from ..scoring import cosine_similarity

logger = logging.getLogger(__name__)

# Placeholder query used to fetch top-k candidates from the store.
# The retriever re-embeds this text; the already-computed embedding is used only for the
# final cosine re-scoring step. This double-embed is acceptable in V1; a future optimisation
# can add a search_by_embedding protocol to skip it.
_DEDUP_PLACEHOLDER = "__dedup__"


class DedupChecker:
    """Checks whether incoming content is a near-duplicate of an existing memory."""

    def __init__(self, retrieval: Retriever, threshold: float) -> None:
        self.retrieval = retrieval
        self.threshold = threshold

    async def find_duplicate(
        self,
        embedding: list[float],
        content: str,
        user_id: str | None,
        limit: int = 3,
    ) -> tuple[MemoryItem, float] | None:
        """Return the best-matching existing memory and its cosine similarity if above threshold."""
        if not embedding:
            return None
        candidates = await self.retrieval.recall(
            query=content,
            limit=limit,
            scoring=False,
            update_access=False,
        )

        best: tuple[MemoryItem, float] | None = None
        for candidate in candidates:
            if user_id is not None and candidate.user_id != user_id:
                continue
            if not candidate.embedding:
                continue
            score = cosine_similarity(embedding, candidate.embedding)
            if score >= self.threshold and (best is None or score > best[1]):
                best = (candidate, score)

        return best
