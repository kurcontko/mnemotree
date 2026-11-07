"""Typed JSON configuration shared by the CLIs; ``Config.open`` opens an engine from it.

Provider settings left out (or ``null``) fall back to the provider's own defaults,
so those defaults live in one place. Keys never go in the config: ``api_key_env``
names the environment variable that holds them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .core.models import Record, SearchOptions
from .core.protocols import Embedder, Extractor
from .engine import EvidenceEngine
from .errors import ConfigurationError
from .retrieval.packing import TiktokenCounter


def secret(name: str) -> str:
    """A credential from the environment; a missing one names the variable to set."""
    try:
        return os.environ[name]
    except KeyError:
        raise ConfigurationError(f"set the {name} environment variable") from None


class ProviderConfig(Record):
    """One provider's settings; ``kind`` selects the implementation."""

    @model_validator(mode="before")
    @classmethod
    def no_inline_secret(cls, data: Any) -> Any:
        if isinstance(data, dict) and "api_key" in data:
            raise ValueError("use api_key_env instead of storing a secret in configuration")
        return data

    def provider_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"kind", "api_key_env"}, exclude_none=True)


class OpenAIEmbeddingConfig(ProviderConfig):
    kind: Literal["openai"]
    model: str
    dimensions: int
    revision: str
    api_key_env: str = "EMBEDDING_API_KEY"
    base_url: str | None = None
    batch_size: int | None = None
    timeout: float | None = None
    # False if the endpoint does not accept the ``dimensions`` parameter.
    send_dimensions: bool | None = None
    max_concurrency: int | None = None
    max_attempts: int | None = None
    backoff_seconds: float | None = None
    max_total_wait: float | None = None

    def build(self) -> Embedder:
        from .providers.embeddings import OpenAIEmbedder

        return OpenAIEmbedder(api_key=secret(self.api_key_env), **self.provider_kwargs())


class SentenceTransformerEmbeddingConfig(ProviderConfig):
    kind: Literal["sentence-transformers"]
    model: str
    revision: str
    query_prompt: str | None = None
    document_prompt: str | None = None

    def build(self) -> Embedder:
        from .providers.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(**self.provider_kwargs())


class FastEmbedEmbeddingConfig(ProviderConfig):
    kind: Literal["fastembed"]
    model: str
    source: str
    revision: str
    dimensions: int
    cache_dir: str | None = None
    query_prompt: str | None = None
    document_prompt: str | None = None
    batch_size: int | None = None

    def build(self) -> Embedder:
        from .providers.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(**self.provider_kwargs())


EmbeddingConfig = Annotated[
    OpenAIEmbeddingConfig | SentenceTransformerEmbeddingConfig | FastEmbedEmbeddingConfig,
    Field(discriminator="kind"),
]


class OpenAIExtractionConfig(ProviderConfig):
    kind: Literal["openai"]
    model: str
    revision: str
    api_key_env: str = "EXTRACTION_API_KEY"
    base_url: str | None = None
    timeout: float | None = None

    def build(self) -> Extractor:
        from .providers.extraction import OpenAIExtractor

        return OpenAIExtractor(api_key=secret(self.api_key_env), **self.provider_kwargs())


class Config(Record):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    tokenizer: str = "o200k_base"
    embedding: EmbeddingConfig | None = None
    extraction: OpenAIExtractionConfig | None = None
    extraction_batch_chars: int = 12000
    search: SearchOptions | None = None

    def open(self, path: Path) -> EvidenceEngine:
        """Open the engine at ``path`` with the providers this config selects.

        A provider credential missing from the environment raises ConfigurationError.
        """
        extractor = self.extraction.build() if self.extraction is not None else None
        embedder = self.embedding.build() if self.embedding is not None else None
        return EvidenceEngine(
            path,
            embedder=embedder,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            token_counter=TiktokenCounter(self.tokenizer),
            extractor=extractor,
            extraction_batch_chars=self.extraction_batch_chars,
        )
