"""Provider boundaries; no agent framework or legacy MemoryItem dependency."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import EmbeddingSpec


class Embedder(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    async def embed_query(self, text: str) -> Sequence[float]: ...


class Reranker(Protocol):
    async def score(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


class TokenCounter(Protocol):
    @property
    def signature(self) -> str: ...

    def count(self, text: str) -> int: ...
