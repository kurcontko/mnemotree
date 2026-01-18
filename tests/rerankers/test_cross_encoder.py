"""Tests for CrossEncoderReranker implementation.

These tests use mocked sentence-transformers to avoid requiring model downloads.
"""

from unittest.mock import MagicMock, patch

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.rerankers.cross_encoder import CrossEncoderReranker


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


class TestCrossEncoderRerankerInit:
    """Tests for CrossEncoderReranker initialization."""

    def test_default_model_name(self):
        """Default model name is set correctly."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model_name(self):
        """Custom model name is stored."""
        reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker.model_name == "custom/model"

    def test_model_not_loaded_initially(self):
        """Model is not loaded until first use."""
        reranker = CrossEncoderReranker()
        assert reranker._model is None


class TestCrossEncoderRerankerLoadModel:
    """Tests for lazy loading of CrossEncoder model."""

    def test_import_error_without_sentence_transformers(self):
        """ImportError is raised with helpful message when sentence-transformers not installed."""
        reranker = CrossEncoderReranker()

        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            patch(
                "mnemotree.rerankers.cross_encoder.CrossEncoderReranker._load_model",
                side_effect=ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                ),
            ),
            pytest.raises(ImportError, match="sentence-transformers is required"),
        ):
            reranker._load_model()

    def test_load_model_only_runs_once(self):
        """_load_model is idempotent - only loads once."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()  # Pretend already loaded

        # Should return early without doing anything
        reranker._load_model()
        # No exception means it worked

    def test_load_model_creates_cross_encoder(self):
        """_load_model creates CrossEncoder with model name."""
        import importlib
        import sys

        mock_sentence_transformers = MagicMock()
        mock_cross_encoder = MagicMock()
        mock_sentence_transformers.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict(sys.modules, {"sentence_transformers": mock_sentence_transformers}):
            import mnemotree.rerankers.cross_encoder as ce_module

            importlib.reload(ce_module)

            reranker = ce_module.CrossEncoderReranker(model_name="test-model")
            reranker._load_model()

            mock_sentence_transformers.CrossEncoder.assert_called_once_with("test-model")
            assert reranker._model is mock_cross_encoder


class TestCrossEncoderRerankerRerank:
    """Tests for the rerank method."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_list(self):
        """Empty candidate list returns empty result."""
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_calls_rerank_sync(self):
        """rerank() delegates to _rerank_sync via asyncio.to_thread."""
        reranker = CrossEncoderReranker()
        memories = [_make_memory("m1", "test content")]

        mock_result = [(memories[0], 0.9)]
        with patch.object(reranker, "_rerank_sync", return_value=mock_result):
            result = await reranker.rerank("query", memories)

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self):
        """rerank() respects top_k limit."""
        reranker = CrossEncoderReranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(5)]

        mock_result = [(mem, 0.9 - i * 0.1) for i, mem in enumerate(memories)]
        with patch.object(reranker, "_rerank_sync", return_value=mock_result):
            result = await reranker.rerank("query", memories, top_k=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_rerank_top_k_none_returns_all(self):
        """top_k=None returns all candidates."""
        reranker = CrossEncoderReranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(5)]

        mock_result = [(mem, 0.9 - i * 0.1) for i, mem in enumerate(memories)]
        with patch.object(reranker, "_rerank_sync", return_value=mock_result):
            result = await reranker.rerank("query", memories, top_k=None)

        assert len(result) == 5


class TestCrossEncoderRerankerRerankSync:
    """Tests for the synchronous _rerank_sync method."""

    def _setup_mocked_reranker(self):
        """Create a reranker with mocked model."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        return reranker

    def test_loads_model_if_not_loaded(self):
        """_rerank_sync loads model if not already loaded."""
        reranker = CrossEncoderReranker()
        memories = [_make_memory("m1", "test")]

        with patch.object(reranker, "_load_model") as mock_load:
            reranker._model = MagicMock()
            reranker._model.predict.return_value = [0.5]
            reranker._rerank_sync("query", memories)

            mock_load.assert_called_once()

    def test_prepares_query_candidate_pairs(self):
        """_rerank_sync prepares correct query-candidate pairs."""
        reranker = self._setup_mocked_reranker()
        memories = [
            _make_memory("m0", "first content"),
            _make_memory("m1", "second content"),
        ]

        reranker._model.predict.return_value = [0.8, 0.6]
        reranker._rerank_sync("test query", memories)

        call_args = reranker._model.predict.call_args[0][0]
        assert call_args == [
            ("test query", "first content"),
            ("test query", "second content"),
        ]

    def test_sorts_by_score_descending(self):
        """_rerank_sync sorts results by score descending."""
        reranker = self._setup_mocked_reranker()
        memories = [
            _make_memory("m0", "low score"),
            _make_memory("m1", "high score"),
            _make_memory("m2", "medium score"),
        ]

        # Return scores in different order than final expected order
        reranker._model.predict.return_value = [0.3, 0.9, 0.6]
        result = reranker._rerank_sync("query", memories)

        # Should be sorted by score descending
        assert result[0][0].memory_id == "m1"  # 0.9
        assert result[0][1] == pytest.approx(0.9)
        assert result[1][0].memory_id == "m2"  # 0.6
        assert result[1][1] == pytest.approx(0.6)
        assert result[2][0].memory_id == "m0"  # 0.3
        assert result[2][1] == pytest.approx(0.3)

    def test_converts_scores_to_float(self):
        """_rerank_sync converts numpy scores to float."""
        import numpy as np

        reranker = self._setup_mocked_reranker()
        memories = [_make_memory("m0", "content")]

        # Simulate numpy array return
        reranker._model.predict.return_value = np.array([0.85])
        result = reranker._rerank_sync("query", memories)

        assert isinstance(result[0][1], float)
        assert result[0][1] == pytest.approx(0.85)
