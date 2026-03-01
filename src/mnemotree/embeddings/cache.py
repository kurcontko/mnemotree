"""Cached embeddings wrapper with TTL-based LRU cache."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .._protocols import AsyncEmbeddingModel


@dataclass
class CacheEntry:
    """Single cache entry with expiration timestamp."""

    embedding: list[float]
    expires_at: float


class CachedEmbeddings:
    """Wrapper that caches embeddings with TTL-based LRU eviction.

    This reduces latency for repeated content by avoiding redundant
    embedding computations. Cache uses content hash as key.

    Args:
        embedder: The underlying embeddings model to wrap.
        max_size: Maximum number of entries to cache (default: 1000).
        ttl_seconds: Time-to-live for cache entries in seconds (default: 3600).

    Example:
        >>> from mnemotree.embeddings.local import LocalSentenceTransformerEmbeddings
        >>> base = LocalSentenceTransformerEmbeddings()
        >>> cached = CachedEmbeddings(base, max_size=500, ttl_seconds=1800)
        >>> # First call computes embedding
        >>> emb1 = await cached.aembed_query("Hello world")
        >>> # Second call returns cached result (fast)
        >>> emb2 = await cached.aembed_query("Hello world")
    """

    def __init__(
        self,
        embedder: AsyncEmbeddingModel,
        *,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._embedder = embedder
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a fraction (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute stable hash key for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        return time.monotonic() > entry.expires_at

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        now = time.monotonic()
        expired_keys = [k for k, v in self._cache.items() if now > v.expires_at]
        for key in expired_keys:
            del self._cache[key]

    def _get_cached(self, text: str) -> list[float] | None:
        """Get cached embedding if valid, otherwise None."""
        key = self._hash_text(text)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry.embedding

    def _put_cached(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache with TTL."""
        key = self._hash_text(text)
        expires_at = time.monotonic() + self._ttl_seconds
        self._cache[key] = CacheEntry(embedding=embedding, expires_at=expires_at)
        self._cache.move_to_end(key)

        # Evict oldest entries if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def embed_query(self, text: str) -> list[float]:
        """Embed query text with caching (synchronous)."""
        cached = self._get_cached(text)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        embedding = self._embedder.embed_query(text)
        self._put_cached(text, embedding)
        return embedding

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text with caching (asynchronous)."""
        async with self._lock:
            cached = self._get_cached(text)
            if cached is not None:
                self._hits += 1
                return cached
            self._misses += 1

        embedding = await self._embedder.aembed_query(text)

        async with self._lock:
            self._put_cached(text, embedding)

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents with caching (synchronous)."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                self._hits += 1
                results[i] = cached
            else:
                self._misses += 1
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self._embedder.embed_documents(uncached_texts)
            for idx, text, embedding in zip(
                uncached_indices, uncached_texts, new_embeddings, strict=True
            ):
                self._put_cached(text, embedding)
                results[idx] = embedding

        # All slots should be filled; cast type (filtering would misalign indices)
        return list(results)  # type: ignore[arg-type]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents with caching (asynchronous)."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        async with self._lock:
            for i, text in enumerate(texts):
                cached = self._get_cached(text)
                if cached is not None:
                    self._hits += 1
                    results[i] = cached
                else:
                    self._misses += 1
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = await self._embedder.aembed_documents(uncached_texts)
            async with self._lock:
                for idx, text, embedding in zip(
                    uncached_indices, uncached_texts, new_embeddings, strict=True
                ):
                    self._put_cached(text, embedding)
                    results[idx] = embedding

        # All slots should be filled; cast type (filtering would misalign indices)
        return list(results)  # type: ignore[arg-type]
