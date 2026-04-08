"""Internal protocols for embeddings and language models.

These protocols decouple mnemotree core from any specific LLM/embedding
framework (LangChain, PydanticAI, etc.).  Implementations can be thin
wrappers around any backend: sentence-transformers, Ollama, vLLM, OpenAI,
or the existing LangChain classes.

``langchain_core.embeddings.Embeddings`` and
``langchain_core.language_models.base.BaseLanguageModel`` both satisfy
these protocols at runtime, so existing code continues to work unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Minimal interface for embedding text into vectors."""

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text asynchronously."""
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents asynchronously."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text synchronously."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents synchronously."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface for invoking a language model."""

    async def ainvoke(self, prompt: str | Any) -> Any:
        """Invoke the LLM asynchronously.

        The return value should have a ``.content`` attribute with the text
        response (matching LangChain's ``AIMessage`` contract).
        """
        ...
