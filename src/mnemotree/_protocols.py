"""
Internal protocols that decouple the core from any specific LLM/embedding library.
These are intentionally minimal — only what Mnemotree actually uses.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Synchronous embedding interface, compatible with LangChain Embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class AsyncEmbeddingModel(EmbeddingModel, Protocol):
    """Async embedding interface."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal LLM interface — just structured async invocation."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any: ...
