"""Tests for local embeddings module.

Tests the LocalSentenceTransformerEmbeddings class with mocked
SentenceTransformer to avoid downloading models in CI.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLocalSentenceTransformerEmbeddings:
    """Tests for LocalSentenceTransformerEmbeddings."""

    def test_initialization(self):
        """Embeddings class initializes with model."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings(
                model_name="test-model",
                device="cpu",
                normalize=True,
                batch_size=16,
            )

            mock_st.SentenceTransformer.assert_called_once_with("test-model", device="cpu")
            assert embeddings.normalize is True
            assert embeddings.batch_size == 16

    def test_embed_documents(self):
        """embed_documents returns list of vectors."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings()
            result = embeddings.embed_documents(["Hello", "World"])

            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]

    def test_embed_documents_empty_list(self):
        """embed_documents returns empty list for empty input."""
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = MagicMock()

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings()
            result = embeddings.embed_documents([])

            assert result == []

    def test_embed_query(self):
        """embed_query returns single vector."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings()
            result = embeddings.embed_query("Test query")

            assert result == [0.5, 0.6, 0.7]

    def test_batch_size_passed_to_encode(self):
        """Batch size is passed to model.encode()."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings(batch_size=64)
            embeddings.embed_documents(["Test"])

            mock_model.encode.assert_called_once()
            call_kwargs = mock_model.encode.call_args[1]
            assert call_kwargs["batch_size"] == 64

    def test_normalize_option(self):
        """normalize_embeddings is passed to encode when supported."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings(normalize=True)
            embeddings.embed_documents(["Test"])

            call_kwargs = mock_model.encode.call_args[1]
            assert call_kwargs.get("normalize_embeddings") is True

    def test_fallback_normalization_on_type_error(self):
        """Falls back to manual normalization if model doesn't support it."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        # First call raises TypeError (old model API), second succeeds
        mock_model.encode.side_effect = [
            TypeError("normalize_embeddings not supported"),
            np.array([[3.0, 4.0]]),  # Vector with norm 5
        ]
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings(normalize=True)
            result = embeddings.embed_documents(["Test"])

            # Should normalize to unit vector: [0.6, 0.8]
            assert len(result) == 1
            assert abs(result[0][0] - 0.6) < 0.01
            assert abs(result[0][1] - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_aembed_query(self):
        """aembed_query runs embed_query in thread."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings()
            result = await embeddings.aembed_query("Async test")

            assert result == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_aembed_documents(self):
        """aembed_documents runs embed_documents in thread."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0], [2.0]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            embeddings = local_module.LocalSentenceTransformerEmbeddings()
            result = await embeddings.aembed_documents(["A", "B"])

            assert len(result) == 2

    def test_import_error_when_sentence_transformers_missing(self):
        """RuntimeError raised when sentence-transformers not installed."""
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            import mnemotree.embeddings.local as local_module
            importlib.reload(local_module)

            with pytest.raises(RuntimeError, match="sentence-transformers"):
                local_module.LocalSentenceTransformerEmbeddings()
