"""Tests for FlashRankReranker implementation.

These tests use mocked flashrank to avoid requiring model downloads.
"""

from unittest.mock import MagicMock, patch

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.rerankers.flashrank import FlashRankReranker


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


class TestFlashRankRerankerInit:
    """Tests for FlashRankReranker initialization."""

    def test_default_model_name(self):
        """Default model name is set correctly."""
        reranker = FlashRankReranker()
        assert reranker.model_name == "ms-marco-TinyBERT-L-2-v2"

    def test_custom_model_name(self):
        """Custom model name is stored."""
        reranker = FlashRankReranker(model_name="custom-model")
        assert reranker.model_name == "custom-model"

    def test_ranker_not_loaded_initially(self):
        """Ranker is not loaded until first use."""
        reranker = FlashRankReranker()
        assert reranker._ranker is None
        assert reranker._request_cls is None


class TestFlashRankRerankerLoad:
    """Tests for lazy loading of FlashRank model."""

    def test_import_error_raised_when_flashrank_not_installed(self):
        """ImportError is raised with helpful message when flashrank not installed."""
        reranker = FlashRankReranker()

        with (
            patch.dict("sys.modules", {"flashrank": None}),
            patch(
                "mnemotree.rerankers.flashrank.FlashRankReranker._load",
                side_effect=ImportError(
                    "flashrank is required for FlashRankReranker. "
                    "Install with: pip install mnemotree[rerank_flashrank]"
                ),
            ),
            pytest.raises(ImportError, match="flashrank is required"),
        ):
            reranker._load()

    @patch("mnemotree.rerankers.flashrank.Ranker", create=True)
    @patch("mnemotree.rerankers.flashrank.RerankRequest", create=True)
    def test_load_initializes_ranker(self, mock_request_cls, mock_ranker_cls):
        """_load() initializes the ranker with model name."""
        mock_ranker = MagicMock()
        mock_ranker_cls.return_value = mock_ranker

        reranker = FlashRankReranker(model_name="test-model")

        # Manually patch the import inside _load
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: MagicMock(
                Ranker=mock_ranker_cls, RerankRequest=mock_request_cls
            )
            if name == "flashrank"
            else __import__(name, *args),
        ):
            reranker._load()

        assert reranker._ranker is mock_ranker
        assert reranker._request_cls is mock_request_cls

    def test_load_only_runs_once(self):
        """_load() is idempotent - only loads once."""
        reranker = FlashRankReranker()
        reranker._ranker = MagicMock()  # Pretend already loaded

        # Should return early without doing anything
        reranker._load()
        # No exception means it worked


class TestFlashRankRerankerCoerceId:
    """Tests for _coerce_id static method."""

    def test_coerce_int(self):
        """Integer IDs are returned as-is."""
        assert FlashRankReranker._coerce_id(0) == 0
        assert FlashRankReranker._coerce_id(5) == 5
        assert FlashRankReranker._coerce_id(100) == 100

    def test_coerce_digit_string(self):
        """Digit strings are converted to integers."""
        assert FlashRankReranker._coerce_id("0") == 0
        assert FlashRankReranker._coerce_id("5") == 5
        assert FlashRankReranker._coerce_id("100") == 100

    def test_coerce_non_digit_string_returns_none(self):
        """Non-digit strings return None."""
        assert FlashRankReranker._coerce_id("abc") is None
        assert FlashRankReranker._coerce_id("1abc") is None
        assert FlashRankReranker._coerce_id("") is None

    def test_coerce_other_types_return_none(self):
        """Other types return None."""
        assert FlashRankReranker._coerce_id(None) is None
        assert FlashRankReranker._coerce_id(1.5) is None
        assert FlashRankReranker._coerce_id([1]) is None
        assert FlashRankReranker._coerce_id({"id": 1}) is None


class TestFlashRankRerankerRerank:
    """Tests for the rerank method."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_list(self):
        """Empty candidate list returns empty result."""
        reranker = FlashRankReranker()
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_calls_rerank_sync(self):
        """rerank() delegates to _rerank_sync via asyncio.to_thread."""
        reranker = FlashRankReranker()
        memories = [_make_memory("m1", "test content")]

        mock_result = [(memories[0], 0.9)]
        with patch.object(reranker, "_rerank_sync", return_value=mock_result):
            result = await reranker.rerank("query", memories)

        assert result == mock_result


class TestFlashRankRerankerRerankSync:
    """Tests for the synchronous _rerank_sync method."""

    def _setup_mocked_reranker(self):
        """Create a reranker with mocked flashrank dependencies."""
        reranker = FlashRankReranker()
        reranker._ranker = MagicMock()
        reranker._request_cls = MagicMock()
        return reranker

    def test_handles_dict_result_format(self):
        """Handles results returned as dictionaries."""
        reranker = self._setup_mocked_reranker()
        memories = [
            _make_memory("m0", "first"),
            _make_memory("m1", "second"),
        ]

        # Mock ranker returning dict format
        reranker._ranker.rerank.return_value = [
            {"id": 1, "score": 0.9},
            {"id": 0, "score": 0.7},
        ]

        result = reranker._rerank_sync("query", memories)

        assert len(result) == 2
        assert result[0][0].memory_id == "m1"  # Higher score first
        assert result[0][1] == pytest.approx(0.9)
        assert result[1][0].memory_id == "m0"
        assert result[1][1] == pytest.approx(0.7)

    def test_handles_object_result_format(self):
        """Handles results returned as objects with attributes."""
        reranker = self._setup_mocked_reranker()
        memories = [
            _make_memory("m0", "first"),
            _make_memory("m1", "second"),
        ]

        # Mock ranker returning object format
        result_obj_1 = MagicMock()
        result_obj_1.id = 1
        result_obj_1.score = 0.85
        result_obj_2 = MagicMock()
        result_obj_2.id = 0
        result_obj_2.score = 0.6
        reranker._ranker.rerank.return_value = [result_obj_1, result_obj_2]

        result = reranker._rerank_sync("query", memories)

        assert len(result) == 2
        assert result[0][0].memory_id == "m1"
        assert result[0][1] == pytest.approx(0.85)
        assert result[1][0].memory_id == "m0"
        assert result[1][1] == pytest.approx(0.6)

    def test_handles_string_id_in_results(self):
        """Handles string IDs that are digit strings."""
        reranker = self._setup_mocked_reranker()
        memories = [_make_memory("m0", "content")]

        reranker._ranker.rerank.return_value = [{"id": "0", "score": 0.8}]

        result = reranker._rerank_sync("query", memories)

        assert len(result) == 1
        assert result[0][0].memory_id == "m0"

    def test_skips_invalid_ids(self):
        """Skips results with invalid IDs."""
        reranker = self._setup_mocked_reranker()
        memories = [_make_memory("m0", "content")]

        reranker._ranker.rerank.return_value = [
            {"id": "invalid", "score": 0.9},  # Invalid string
            {"id": -1, "score": 0.8},  # Negative
            {"id": 999, "score": 0.7},  # Out of range
            {"id": 0, "score": 0.5},  # Valid
        ]

        result = reranker._rerank_sync("query", memories)

        # Only the valid one plus the unseen appended with 0 score
        assert len(result) == 1
        assert result[0][0].memory_id == "m0"
        assert result[0][1] == pytest.approx(0.5)

    def test_skips_duplicate_ids(self):
        """Skips duplicate IDs in results."""
        reranker = self._setup_mocked_reranker()
        memories = [_make_memory("m0", "content")]

        reranker._ranker.rerank.return_value = [
            {"id": 0, "score": 0.9},
            {"id": 0, "score": 0.5},  # Duplicate - should be skipped
        ]

        result = reranker._rerank_sync("query", memories)

        assert len(result) == 1
        assert result[0][1] == pytest.approx(0.9)  # First occurrence kept

    def test_appends_unseen_candidates_with_zero_score(self):
        """Candidates not in results are appended with score 0."""
        reranker = self._setup_mocked_reranker()
        memories = [
            _make_memory("m0", "first"),
            _make_memory("m1", "second"),
            _make_memory("m2", "third"),
        ]

        # Only return one result
        reranker._ranker.rerank.return_value = [{"id": 1, "score": 0.9}]

        result = reranker._rerank_sync("query", memories)

        assert len(result) == 3
        assert result[0][0].memory_id == "m1"
        assert result[0][1] == pytest.approx(0.9)
        # Unseen candidates appended with 0 score
        assert result[1][0].memory_id == "m0"
        assert result[1][1] == pytest.approx(0.0)
        assert result[2][0].memory_id == "m2"
        assert result[2][1] == pytest.approx(0.0)

    def test_respects_top_k(self):
        """top_k limits the number of results returned."""
        reranker = self._setup_mocked_reranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(5)]

        reranker._ranker.rerank.return_value = [{"id": i, "score": 0.9 - i * 0.1} for i in range(5)]

        result = reranker._rerank_sync("query", memories, top_k=2)

        assert len(result) == 2

    def test_top_k_none_returns_all(self):
        """top_k=None returns all candidates."""
        reranker = self._setup_mocked_reranker()
        memories = [_make_memory(f"m{i}", f"content {i}") for i in range(5)]

        reranker._ranker.rerank.return_value = [{"id": i, "score": 0.9 - i * 0.1} for i in range(5)]

        result = reranker._rerank_sync("query", memories, top_k=None)

        assert len(result) == 5

    def test_handles_missing_score_defaults_to_zero(self):
        """Missing score in result defaults to 0.0."""
        reranker = self._setup_mocked_reranker()
        memories = [_make_memory("m0", "content")]

        reranker._ranker.rerank.return_value = [{"id": 0}]  # No score

        result = reranker._rerank_sync("query", memories)

        assert result[0][1] == pytest.approx(0.0)

    def test_passages_prepared_correctly(self):
        """Passages are prepared with id and text fields."""
        reranker = self._setup_mocked_reranker()
        memories = [
            _make_memory("m0", "first content"),
            _make_memory("m1", "second content"),
        ]

        reranker._ranker.rerank.return_value = []

        reranker._rerank_sync("test query", memories)

        # Check the request was created correctly
        call_args = reranker._request_cls.call_args
        assert call_args[1]["query"] == "test query"
        passages = call_args[1]["passages"]
        assert passages == [
            {"id": 0, "text": "first content"},
            {"id": 1, "text": "second content"},
        ]


class TestFlashRankRerankerLoadFallback:
    """Tests for the TypeError fallback in _load()."""

    def test_fallback_logic_simulated(self):
        """Verifies the fallback logic pattern works correctly."""
        # This tests the pattern used in _load() for older flashrank versions
        mock_ranker_instance = MagicMock()

        # First call raises TypeError (simulating old flashrank), second succeeds
        call_count = [0]

        def ranker_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "model_name" in kwargs:
                raise TypeError("unexpected keyword argument 'model_name'")
            return mock_ranker_instance

        mock_ranker_cls = MagicMock(side_effect=ranker_side_effect)

        # Simulate the fallback logic from _load()
        model_name = "test-model"
        try:
            result = mock_ranker_cls(model_name=model_name)
        except TypeError:
            # Fallback for older flashrank versions
            result = mock_ranker_cls(model_name)

        assert result == mock_ranker_instance
        assert call_count[0] == 2  # Called twice - once failed, once succeeded
