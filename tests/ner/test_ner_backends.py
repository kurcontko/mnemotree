"""Tests for NER backend implementations.

These tests use mocked dependencies to avoid requiring heavy model downloads.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemotree.ner import create_ner
from mnemotree.ner.base import BaseNER, NERResult
from mnemotree.ner.composite import CompositeNER


class TestNERResult:
    """Tests for NERResult model."""

    def test_creation_minimal(self):
        """NERResult can be created with minimal fields."""
        result = NERResult(entities={"John": "PERSON"}, mentions={"John": ["Hello John"]})
        assert result.entities == {"John": "PERSON"}
        assert result.mentions == {"John": ["Hello John"]}
        assert result.confidence is None

    def test_creation_with_confidence(self):
        """NERResult can include confidence scores."""
        result = NERResult(
            entities={"Apple": "ORG"},
            mentions={"Apple": ["Apple Inc."]},
            confidence={"Apple": 0.95},
        )
        assert result.confidence == {"Apple": 0.95}


class TestCreateNerFactory:
    """Tests for create_ner() factory function."""

    def test_unknown_backend_raises(self):
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown NER backend"):
            create_ner("nonexistent_backend")

    @patch("mnemotree.ner.SpacyNER")
    def test_spacy_backend(self, mock_spacy):
        """spacy backend creates SpacyNER."""
        create_ner("spacy", model="en_core_web_sm")
        mock_spacy.assert_called_once_with(model="en_core_web_sm")

    @patch("mnemotree.ner.GLiNERNER")
    def test_gliner_backend(self, mock_gliner):
        """gliner backend creates GLiNERNER."""
        create_ner("gliner", model_name="test-model")
        mock_gliner.assert_called_once_with(model_name="test-model")

    @patch("mnemotree.ner.TransformersNER")
    def test_transformers_backend(self, mock_tf):
        """transformers backend creates TransformersNER."""
        create_ner("transformers", model="test/model")
        mock_tf.assert_called_once_with(model="test/model")

    @patch("mnemotree.ner.TransformersNER")
    def test_hf_alias(self, mock_tf):
        """hf is an alias for transformers."""
        create_ner("hf")
        mock_tf.assert_called_once()

    @patch("mnemotree.ner.TransformersNER")
    def test_huggingface_alias(self, mock_tf):
        """huggingface is an alias for transformers."""
        create_ner("huggingface")
        mock_tf.assert_called_once()

    @patch("mnemotree.ner.StanzaNER")
    def test_stanza_backend(self, mock_stanza):
        """stanza backend creates StanzaNER."""
        create_ner("stanza")
        mock_stanza.assert_called_once()

    @patch("mnemotree.ner.SparkNLPNER")
    def test_spark_backend(self, mock_spark):
        """spark backend creates SparkNLPNER."""
        create_ner("spark")
        mock_spark.assert_called_once()

    @patch("mnemotree.ner.SparkNLPNER")
    def test_spark_nlp_alias(self, mock_spark):
        """spark-nlp is an alias for spark."""
        create_ner("spark-nlp")
        mock_spark.assert_called_once()

    @patch("mnemotree.ner.LangchainLLMNER")
    def test_llm_backend(self, mock_llm):
        """llm backend creates LangchainLLMNER."""
        create_ner("llm")
        mock_llm.assert_called_once()

    @patch("mnemotree.ner.LangchainLLMNER")
    def test_langchain_alias(self, mock_llm):
        """langchain is an alias for llm."""
        create_ner("langchain")
        mock_llm.assert_called_once()

    def test_case_insensitive(self):
        """Backend names are case-insensitive."""
        with patch("mnemotree.ner.SpacyNER") as mock_spacy:
            create_ner("SPACY")
            mock_spacy.assert_called_once()

    def test_whitespace_stripped(self):
        """Whitespace is stripped from backend names."""
        with patch("mnemotree.ner.SpacyNER") as mock_spacy:
            create_ner("  spacy  ")
            mock_spacy.assert_called_once()


class TestBaseNERGetContext:
    """Tests for BaseNER._get_context() helper method."""

    class ConcreteNER(BaseNER):
        """Concrete implementation for testing."""

        async def extract_entities(self, text: str) -> NERResult:
            return NERResult(entities={}, mentions={})

    def test_context_middle_of_text(self):
        """Context extraction from middle of text."""
        ner = self.ConcreteNER()
        text = "0" * 100 + "ENTITY" + "0" * 100
        # Entity is at position 100-106
        context = ner._get_context(text, 100, 106, window=20)
        assert len(context) == 20 + 6 + 20  # 20 before, 6 entity, 20 after

    def test_context_at_start(self):
        """Context at start of text doesn't go negative."""
        ner = self.ConcreteNER()
        text = "ENTITY" + "0" * 100
        context = ner._get_context(text, 0, 6, window=20)
        assert context.startswith("ENTITY")

    def test_context_at_end(self):
        """Context at end of text doesn't exceed length."""
        ner = self.ConcreteNER()
        text = "0" * 100 + "ENTITY"
        context = ner._get_context(text, 100, 106, window=20)
        assert context.endswith("ENTITY")


class TestCompositeNER:
    """Tests for CompositeNER."""

    @pytest.mark.asyncio
    async def test_merges_results_from_multiple_backends(self):
        """CompositeNER merges entities from multiple backends."""
        backend1 = MagicMock(spec=BaseNER)
        backend1.extract_entities = AsyncMock(
            return_value=NERResult(
                entities={"John": "PERSON"},
                mentions={"John": ["Hello John"]},
            )
        )

        backend2 = MagicMock(spec=BaseNER)
        backend2.extract_entities = AsyncMock(
            return_value=NERResult(
                entities={"Apple": "ORG"},
                mentions={"Apple": ["Apple Inc"]},
            )
        )

        # CompositeNER takes list of (implementation, weight) tuples
        composite = CompositeNER(implementations=[(backend1, 1.0), (backend2, 1.0)])
        result = await composite.extract_entities("Hello John from Apple")

        # Both entities should be present
        assert "John" in result.entities
        assert "Apple" in result.entities
        assert result.entities["John"] == "PERSON"
        assert result.entities["Apple"] == "ORG"

    @pytest.mark.asyncio
    async def test_handles_overlapping_entities(self):
        """CompositeNER handles same entity from different backends."""
        backend1 = MagicMock(spec=BaseNER)
        backend1.extract_entities = AsyncMock(
            return_value=NERResult(
                entities={"NYC": "GPE"},
                mentions={"NYC": ["from NYC"]},
            )
        )

        backend2 = MagicMock(spec=BaseNER)
        backend2.extract_entities = AsyncMock(
            return_value=NERResult(
                entities={"NYC": "LOC"},  # Different type
                mentions={"NYC": ["in NYC"]},
            )
        )

        # CompositeNER takes list of (implementation, weight) tuples
        composite = CompositeNER(implementations=[(backend1, 1.0), (backend2, 1.0)])
        result = await composite.extract_entities("I went from NYC to NYC")

        # Entity should be present (first wins)
        assert "NYC" in result.entities
        assert result.entities["NYC"] == "GPE"  # First backend wins
        # Mentions should be merged
        assert len(result.mentions.get("NYC", [])) >= 2

