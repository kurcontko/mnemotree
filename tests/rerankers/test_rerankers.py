"""Tests for reranker implementations.

These tests use mocked dependencies to avoid requiring heavy model downloads.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.rerankers import create_reranker
from mnemotree.rerankers.base import BaseReranker, NoOpReranker


def _make_memory(memory_id: str, content: str) -> MemoryItem:
    """Helper to create test MemoryItem instances."""
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp="2024-01-01T00:00:00+00:00",
        embedding=[0.0] * 10,
    )


class TestNoOpReranker:
    """Tests for NoOpReranker."""

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """NoOpReranker preserves original order."""
        reranker = NoOpReranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(3)]

        result = await reranker.rerank("query", memories)

        assert len(result) == 3
        assert result[0][0].memory_id == "m0"
        assert result[1][0].memory_id == "m1"
        assert result[2][0].memory_id == "m2"

    @pytest.mark.asyncio
    async def test_assigns_unit_scores(self):
        """NoOpReranker assigns score of 1.0 to all items."""
        reranker = NoOpReranker()
        memories = [_make_memory("m1", "test")]

        result = await reranker.rerank("query", memories)

        assert result[0][1] == 1.0

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        """NoOpReranker respects top_k limit."""
        reranker = NoOpReranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(10)]

        result = await reranker.rerank("query", memories, top_k=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_top_k_none_returns_all(self):
        """NoOpReranker returns all items when top_k is None."""
        reranker = NoOpReranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(5)]

        result = await reranker.rerank("query", memories, top_k=None)

        assert len(result) == 5


class TestCreateRerankerFactory:
    """Tests for create_reranker() factory function."""

    def test_unknown_backend_raises(self):
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown reranker backend"):
            create_reranker("nonexistent")

    def test_noop_backend(self):
        """noop backend creates NoOpReranker."""
        reranker = create_reranker("noop")
        assert isinstance(reranker, NoOpReranker)

    def test_none_alias(self):
        """none is an alias for noop."""
        reranker = create_reranker("none")
        assert isinstance(reranker, NoOpReranker)

    def test_passthrough_alias(self):
        """passthrough is an alias for noop."""
        reranker = create_reranker("passthrough")
        assert isinstance(reranker, NoOpReranker)

    @patch("mnemotree.rerankers.FlashRankReranker")
    def test_flashrank_backend(self, mock_flashrank):
        """flashrank backend creates FlashRankReranker."""
        create_reranker("flashrank", model_name="test-model")
        mock_flashrank.assert_called_once_with(model_name="test-model")

    @patch("mnemotree.rerankers.FlashRankReranker")
    def test_flash_rank_alias(self, mock_flashrank):
        """flash-rank is an alias for flashrank."""
        create_reranker("flash-rank")
        mock_flashrank.assert_called_once()

    @patch("mnemotree.rerankers.CrossEncoderReranker")
    def test_cross_encoder_backend(self, mock_ce):
        """cross-encoder backend creates CrossEncoderReranker."""
        create_reranker("cross-encoder", model_name="test-model")
        mock_ce.assert_called_once_with(model_name="test-model")

    @patch("mnemotree.rerankers.CrossEncoderReranker")
    def test_ce_alias(self, mock_ce):
        """ce is an alias for cross-encoder."""
        create_reranker("ce")
        mock_ce.assert_called_once()

    def test_case_insensitive(self):
        """Backend names are case-insensitive."""
        reranker = create_reranker("NOOP")
        assert isinstance(reranker, NoOpReranker)

    def test_whitespace_stripped(self):
        """Whitespace is stripped from backend names."""
        reranker = create_reranker("  noop  ")
        assert isinstance(reranker, NoOpReranker)


class TestBaseRerankerInterface:
    """Tests for BaseReranker interface compliance."""

    class ConcreteReranker(BaseReranker):
        """Concrete implementation for testing."""

        async def rerank(
            self, query: str, candidates: list[MemoryItem], top_k: int | None = None
        ) -> list[tuple[MemoryItem, float]]:
            # Simple mock implementation - reverse order with decreasing scores
            results = [(mem, 1.0 - i * 0.1) for i, mem in enumerate(candidates)]
            if top_k is not None:
                results = results[:top_k]
            return results

    @pytest.mark.asyncio
    async def test_rerank_returns_tuples(self):
        """rerank() returns list of (MemoryItem, float) tuples."""
        reranker = self.ConcreteReranker()
        memories = [_make_memory("m1", "test")]

        result = await reranker.rerank("query", memories)

        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], MemoryItem)
        assert isinstance(result[0][1], float)

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """rerank() handles empty candidate list."""
        reranker = self.ConcreteReranker()

        result = await reranker.rerank("query", [])

        assert result == []
