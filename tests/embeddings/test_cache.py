"""Tests for the CachedEmbeddings wrapper."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.embeddings.cache import CachedEmbeddings


@pytest.fixture
def mock_embedder():
    """Create a mock embedder with predictable outputs."""
    embedder = MagicMock()
    embedder.embed_query = MagicMock(side_effect=lambda text: [hash(text) % 100 / 100.0] * 8)
    embedder.aembed_query = AsyncMock(side_effect=lambda text: [hash(text) % 100 / 100.0] * 8)
    embedder.embed_documents = MagicMock(
        side_effect=lambda texts: [[hash(t) % 100 / 100.0] * 8 for t in texts]
    )
    embedder.aembed_documents = AsyncMock(
        side_effect=lambda texts: [[hash(t) % 100 / 100.0] * 8 for t in texts]
    )
    return embedder


class TestCachedEmbeddings:
    """Tests for CachedEmbeddings class."""

    def test_cache_hit_sync(self, mock_embedder):
        """Test that repeated sync calls return cached results."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=60)

        # First call should miss cache
        result1 = cached.embed_query("hello world")
        assert mock_embedder.embed_query.call_count == 1
        assert cached._hits == 0
        assert cached._misses == 1

        # Second call should hit cache
        result2 = cached.embed_query("hello world")
        assert mock_embedder.embed_query.call_count == 1  # Not called again
        assert cached._hits == 1
        assert cached._misses == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_hit_async(self, mock_embedder):
        """Test that repeated async calls return cached results."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=60)

        result1 = await cached.aembed_query("hello async")
        assert mock_embedder.aembed_query.call_count == 1

        result2 = await cached.aembed_query("hello async")
        assert mock_embedder.aembed_query.call_count == 1
        assert result1 == result2
        assert cached.hit_rate == 0.5  # 1 hit, 1 miss

    def test_cache_miss_different_content(self, mock_embedder):
        """Test that different content misses cache."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=60)

        cached.embed_query("text one")
        cached.embed_query("text two")

        assert mock_embedder.embed_query.call_count == 2
        assert cached._misses == 2
        assert cached._hits == 0

    def test_lru_eviction(self, mock_embedder):
        """Test that old entries are evicted when cache is full."""
        cached = CachedEmbeddings(mock_embedder, max_size=3, ttl_seconds=60)

        # Fill cache to capacity
        cached.embed_query("one")
        cached.embed_query("two")
        cached.embed_query("three")
        assert len(cached._cache) == 3

        # Add one more, should evict oldest
        cached.embed_query("four")
        assert len(cached._cache) == 3  # Still at max

        # "one" should be evicted (it was oldest)
        # Accessing "one" should miss
        cached.embed_query("one")
        assert mock_embedder.embed_query.call_count == 5  # Had to recompute

    def test_ttl_expiration(self, mock_embedder):
        """Test that expired entries are not returned."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=0.1)

        cached.embed_query("expiring text")
        assert cached._misses == 1

        # Wait for expiration
        time.sleep(0.15)

        # Should miss again due to expiration
        cached.embed_query("expiring text")
        assert mock_embedder.embed_query.call_count == 2
        assert cached._misses == 2

    def test_embed_documents_caching(self, mock_embedder):
        """Test that batch embedding also uses cache."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=60)

        # Pre-populate cache with one document
        cached.embed_query("doc one")
        assert cached._misses == 1

        # Batch embed with one cached, two new
        results = cached.embed_documents(["doc one", "doc two", "doc three"])
        assert len(results) == 3

        # Should have hit cache for "doc one"
        assert cached._hits == 1
        # Should have missed for "doc two" and "doc three"
        assert cached._misses == 3

    def test_stats_property(self, mock_embedder):
        """Test that stats returns correct values."""
        cached = CachedEmbeddings(mock_embedder, max_size=100, ttl_seconds=60)

        cached.embed_query("test")
        cached.embed_query("test")
        cached.embed_query("other")

        stats = cached.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)
        assert stats["size"] == 2
        assert stats["max_size"] == 100

    def test_clear(self, mock_embedder):
        """Test that clear removes all entries and resets stats."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=60)

        cached.embed_query("test")
        cached.embed_query("test")  # Hit
        assert len(cached._cache) == 1
        assert cached._hits == 1

        cached.clear()
        assert len(cached._cache) == 0
        assert cached._hits == 0
        assert cached._misses == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, mock_embedder):
        """Test that concurrent async access is handled correctly."""
        cached = CachedEmbeddings(mock_embedder, max_size=10, ttl_seconds=60)

        # Run multiple concurrent requests for same content
        tasks = [cached.aembed_query("concurrent") for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be identical
        assert all(r == results[0] for r in results)

        # Should have at most one miss (first caller computes)
        # Due to lock, some may get cache hits
        assert cached._hits + cached._misses == 5
