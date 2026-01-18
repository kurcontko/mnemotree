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


class TestGLiNERNER:
    """Tests for GLiNER NER backend with mocked model."""

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self):
        """GLiNERNER can extract entities with mocked model."""
        import importlib
        import sys

        # Create mock for GLiNER
        mock_gliner = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "John", "label": "person", "score": 0.95, "start": 0, "end": 4},
            {"text": "Apple", "label": "organization", "score": 0.88, "start": 18, "end": 23},
        ]
        mock_gliner.GLiNER.from_pretrained.return_value = mock_model

        # Mock the gliner module before importing
        with patch.dict(sys.modules, {"gliner": mock_gliner}):
            import mnemotree.ner.gliner as gliner_module
            importlib.reload(gliner_module)

            ner = gliner_module.GLiNERNER(model_name="test-model", threshold=0.3)
            result = await ner.extract_entities("John works at Apple Inc.")

            assert "John" in result.entities
            assert result.entities["John"] == "person"
            assert "Apple" in result.entities
            assert result.entities["Apple"] == "organization"
            assert result.confidence["John"] == pytest.approx(0.95)
            assert result.confidence["Apple"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_custom_entity_types(self):
        """GLiNERNER uses custom entity types."""
        import importlib
        import sys

        mock_gliner = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mock_gliner.GLiNER.from_pretrained.return_value = mock_model

        with patch.dict(sys.modules, {"gliner": mock_gliner}):
            import mnemotree.ner.gliner as gliner_module
            importlib.reload(gliner_module)

            custom_types = ["food", "restaurant", "ingredient"]
            ner = gliner_module.GLiNERNER(entity_types=custom_types)
            await ner.extract_entities("Test text")

            mock_model.predict_entities.assert_called_once()
            call_args = mock_model.predict_entities.call_args
            assert call_args[0][1] == custom_types

    @pytest.mark.asyncio
    async def test_threshold_passed_to_model(self):
        """GLiNERNER passes threshold to model."""
        import importlib
        import sys

        mock_gliner = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mock_gliner.GLiNER.from_pretrained.return_value = mock_model

        with patch.dict(sys.modules, {"gliner": mock_gliner}):
            import mnemotree.ner.gliner as gliner_module
            importlib.reload(gliner_module)

            ner = gliner_module.GLiNERNER(threshold=0.7)
            await ner.extract_entities("Test")

            call_kwargs = mock_model.predict_entities.call_args[1]
            assert call_kwargs["threshold"] == pytest.approx(0.7)

    def test_import_error_without_gliner(self):
        """GLiNERNER raises ImportError when gliner not installed."""
        import importlib
        import sys

        with patch.dict(sys.modules, {"gliner": None}):
            import mnemotree.ner.gliner as gliner_module
            importlib.reload(gliner_module)

            with pytest.raises(ImportError, match="GLiNER is not installed"):
                gliner_module.GLiNERNER()


class TestTransformersNER:
    """Tests for TransformersNER backend with mocked pipeline."""

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self):
        """TransformersNER extracts entities from pipeline output."""
        import importlib
        import sys

        mock_transformers = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"word": "Paris", "entity_group": "LOC", "score": 0.92, "start": 10, "end": 15},
            {"word": "France", "entity_group": "LOC", "score": 0.89, "start": 17, "end": 23},
        ]
        mock_transformers.pipeline.return_value = mock_pipe

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            import mnemotree.ner.hf_transformers as hf_module
            importlib.reload(hf_module)

            ner = hf_module.TransformersNER(model="test-model")
            result = await ner.extract_entities("I visited Paris, France last summer")

            assert "Paris" in result.entities
            assert result.entities["Paris"] == "LOC"
            assert "France" in result.entities

    @pytest.mark.asyncio
    async def test_handles_different_output_formats(self):
        """TransformersNER handles various output field names."""
        import importlib
        import sys

        mock_transformers = MagicMock()
        mock_pipe = MagicMock()
        # Some models use "entity" instead of "entity_group"
        mock_pipe.return_value = [
            {"text": "Berlin", "entity": "GPE", "score": 0.85, "start": 0, "end": 6},
        ]
        mock_transformers.pipeline.return_value = mock_pipe

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            import mnemotree.ner.hf_transformers as hf_module
            importlib.reload(hf_module)

            ner = hf_module.TransformersNER()
            result = await ner.extract_entities("Berlin is nice")

            assert "Berlin" in result.entities

    @pytest.mark.asyncio
    async def test_handles_empty_word(self):
        """TransformersNER skips entities with empty text."""
        import importlib
        import sys

        mock_transformers = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"word": "", "entity_group": "LOC", "score": 0.5},
            {"word": "Valid", "entity_group": "PER", "score": 0.9},
        ]
        mock_transformers.pipeline.return_value = mock_pipe

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            import mnemotree.ner.hf_transformers as hf_module
            importlib.reload(hf_module)

            ner = hf_module.TransformersNER()
            result = await ner.extract_entities("Valid text")

            assert "" not in result.entities
            assert "Valid" in result.entities

    @pytest.mark.asyncio
    async def test_higher_score_updates_entity_type(self):
        """Higher confidence score updates entity type."""
        import importlib
        import sys

        mock_transformers = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"word": "ABC", "entity_group": "ORG", "score": 0.6, "start": 0, "end": 3},
            {"word": "ABC", "entity_group": "MISC", "score": 0.9, "start": 10, "end": 13},
        ]
        mock_transformers.pipeline.return_value = mock_pipe

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            import mnemotree.ner.hf_transformers as hf_module
            importlib.reload(hf_module)

            ner = hf_module.TransformersNER()
            result = await ner.extract_entities("ABC test ABC again")

            # Higher score should update type
            assert result.entities["ABC"] == "MISC"
            assert result.confidence["ABC"] == pytest.approx(0.9)

    def test_fallback_to_grouped_entities(self):
        """TransformersNER falls back to grouped_entities if aggregation_strategy fails."""
        import importlib
        import sys

        mock_transformers = MagicMock()
        # First call raises TypeError, second succeeds
        mock_transformers.pipeline.side_effect = [
            TypeError("aggregation_strategy not supported"),
            MagicMock(),
        ]

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            import mnemotree.ner.hf_transformers as hf_module
            importlib.reload(hf_module)

            # Should not raise
            ner = hf_module.TransformersNER()
            assert ner._pipeline is not None

    def test_import_error_without_transformers(self):
        """TransformersNER raises ImportError when transformers not installed."""
        import importlib
        import sys

        with patch.dict(sys.modules, {"transformers": None}):
            import mnemotree.ner.hf_transformers as hf_module
            importlib.reload(hf_module)

            with pytest.raises(ImportError, match="TransformersNER requires"):
                hf_module.TransformersNER()


class TestStanzaNER:
    """Tests for StanzaNER backend with mocked stanza."""

    def test_import_error_without_stanza(self):
        """StanzaNER raises ImportError when stanza not installed."""
        import importlib
        import sys

        with patch.dict(sys.modules, {"stanza": None}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            with pytest.raises(ImportError, match="StanzaNER requires"):
                stanza_module.StanzaNER()

    def test_initialization_with_defaults(self):
        """StanzaNER initializes with default parameters."""
        import importlib
        import sys

        mock_stanza = MagicMock()
        mock_pipeline = MagicMock()
        mock_stanza.Pipeline.return_value = mock_pipeline

        with patch.dict(sys.modules, {"stanza": mock_stanza}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            ner = stanza_module.StanzaNER()

            mock_stanza.Pipeline.assert_called_once_with(
                lang="en",
                processors="tokenize,ner",
                use_gpu=False,
                verbose=False,
            )

    def test_initialization_with_custom_params(self):
        """StanzaNER accepts custom parameters."""
        import importlib
        import sys

        mock_stanza = MagicMock()
        mock_stanza.Pipeline.return_value = MagicMock()

        with patch.dict(sys.modules, {"stanza": mock_stanza}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            stanza_module.StanzaNER(
                lang="de",
                processors="tokenize,ner,pos",
                use_gpu=True,
                pipeline_kwargs={"verbose": True, "download_method": "reuse"},
            )

            call_kwargs = mock_stanza.Pipeline.call_args[1]
            assert call_kwargs["lang"] == "de"
            assert call_kwargs["processors"] == "tokenize,ner,pos"
            assert call_kwargs["use_gpu"] is True
            assert call_kwargs["verbose"] is True  # From pipeline_kwargs
            assert call_kwargs["download_method"] == "reuse"

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self):
        """StanzaNER extracts entities from document."""
        import importlib
        import sys

        mock_stanza = MagicMock()
        mock_pipeline = MagicMock()
        mock_doc = MagicMock()

        # Create mock entities
        mock_ent1 = MagicMock()
        mock_ent1.text = "Microsoft"
        mock_ent1.type = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 9

        mock_ent2 = MagicMock()
        mock_ent2.text = "Seattle"
        mock_ent2.type = "GPE"
        mock_ent2.start_char = 22
        mock_ent2.end_char = 29

        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_pipeline.return_value = mock_doc
        mock_stanza.Pipeline.return_value = mock_pipeline

        with patch.dict(sys.modules, {"stanza": mock_stanza}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            ner = stanza_module.StanzaNER()
            result = await ner.extract_entities("Microsoft is based in Seattle, Washington")

            assert "Microsoft" in result.entities
            assert result.entities["Microsoft"] == "ORG"
            assert "Seattle" in result.entities
            assert result.entities["Seattle"] == "GPE"

    @pytest.mark.asyncio
    async def test_extract_entities_with_label_fallback(self):
        """StanzaNER uses 'label' attribute if 'type' is missing."""
        import importlib
        import sys

        mock_stanza = MagicMock()
        mock_pipeline = MagicMock()
        mock_doc = MagicMock()

        mock_ent = MagicMock(spec=["text", "label", "start_char", "end_char"])
        mock_ent.text = "Paris"
        mock_ent.label = "LOC"  # Note: label instead of type
        mock_ent.start_char = 0
        mock_ent.end_char = 5

        # Simulate missing 'type' attribute
        del mock_ent.type
        mock_doc.ents = [mock_ent]
        mock_pipeline.return_value = mock_doc
        mock_stanza.Pipeline.return_value = mock_pipeline

        with patch.dict(sys.modules, {"stanza": mock_stanza}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            ner = stanza_module.StanzaNER()
            result = await ner.extract_entities("Paris is beautiful")

            assert result.entities.get("Paris") == "LOC"

    @pytest.mark.asyncio
    async def test_extract_entities_skips_empty_text(self):
        """StanzaNER skips entities with empty text."""
        import importlib
        import sys

        mock_stanza = MagicMock()
        mock_pipeline = MagicMock()
        mock_doc = MagicMock()

        mock_ent = MagicMock()
        mock_ent.text = "   "  # Whitespace only
        mock_ent.type = "ORG"

        mock_doc.ents = [mock_ent]
        mock_pipeline.return_value = mock_doc
        mock_stanza.Pipeline.return_value = mock_pipeline

        with patch.dict(sys.modules, {"stanza": mock_stanza}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            ner = stanza_module.StanzaNER()
            result = await ner.extract_entities("Some text")

            assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_no_ents_attribute(self):
        """StanzaNER handles documents without ents attribute."""
        import importlib
        import sys

        mock_stanza = MagicMock()
        mock_pipeline = MagicMock()
        mock_doc = MagicMock(spec=[])  # No ents attribute

        mock_pipeline.return_value = mock_doc
        mock_stanza.Pipeline.return_value = mock_pipeline

        with patch.dict(sys.modules, {"stanza": mock_stanza}):
            import mnemotree.ner.stanza as stanza_module
            importlib.reload(stanza_module)

            ner = stanza_module.StanzaNER()
            result = await ner.extract_entities("No entities here")

            assert result.entities == {}
            assert result.mentions == {}


class TestLangchainLLMNER:
    """Tests for LangchainLLMNER backend."""

    def test_initialization(self):
        """LangchainLLMNER stores LLM and sets up parser."""
        from mnemotree.ner.llm import LangchainLLMNER

        mock_llm = MagicMock()
        ner = LangchainLLMNER(llm=mock_llm)

        assert ner.llm is mock_llm
        assert ner.parser is not None
        assert "entities" in ner.prompt_template

    @pytest.mark.asyncio
    async def test_extract_entities_success(self):
        """LangchainLLMNER extracts entities from LLM response."""
        from mnemotree.ner.llm import LangchainLLMNER

        mock_llm = MagicMock()
        ner = LangchainLLMNER(llm=mock_llm)

        # Mock the chain execution
        with patch.object(ner, "parser") as mock_parser:
            mock_parser.get_format_instructions.return_value = "format instructions"

            # Create a mock chain that returns parsed result
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = {
                "entities": {"Apple": "ORG", "Tim Cook": "PERSON"},
                "mentions": {},
                "confidence": {"Apple": 0.95, "Tim Cook": 0.9},
            }

            with patch("mnemotree.ner.llm.PromptTemplate") as mock_prompt:
                mock_prompt.return_value.__or__ = lambda self, other: MagicMock(
                    __or__=lambda self, other: mock_chain
                )

                result = await ner.extract_entities("Apple CEO Tim Cook announced...")

        assert "Apple" in result.entities
        assert result.entities["Apple"] == "ORG"
        assert "Tim Cook" in result.entities
        assert result.entities["Tim Cook"] == "PERSON"

    @pytest.mark.asyncio
    async def test_extract_entities_validation_error_returns_empty(self):
        """LangchainLLMNER returns empty result on validation error."""
        from pydantic import ValidationError

        from mnemotree.ner.llm import LangchainLLMNER

        mock_llm = MagicMock()
        ner = LangchainLLMNER(llm=mock_llm)

        with patch.object(ner, "parser") as mock_parser:
            mock_parser.get_format_instructions.return_value = "format"

            # Create a mock chain that raises ValidationError
            mock_chain = AsyncMock()
            mock_chain.ainvoke.side_effect = ValidationError.from_exception_data(
                "test", []
            )

            with patch("mnemotree.ner.llm.PromptTemplate") as mock_prompt:
                mock_prompt.return_value.__or__ = lambda self, other: MagicMock(
                    __or__=lambda self, other: mock_chain
                )

                result = await ner.extract_entities("Some text")

        assert result.entities == {}
        assert result.mentions == {}

    @pytest.mark.asyncio
    async def test_extract_entities_type_error_returns_empty(self):
        """LangchainLLMNER returns empty result on TypeError."""
        from mnemotree.ner.llm import LangchainLLMNER

        mock_llm = MagicMock()
        ner = LangchainLLMNER(llm=mock_llm)

        with patch.object(ner, "parser") as mock_parser:
            mock_parser.get_format_instructions.return_value = "format"

            mock_chain = AsyncMock()
            mock_chain.ainvoke.side_effect = TypeError("type error")

            with patch("mnemotree.ner.llm.PromptTemplate") as mock_prompt:
                mock_prompt.return_value.__or__ = lambda self, other: MagicMock(
                    __or__=lambda self, other: mock_chain
                )

                result = await ner.extract_entities("Some text")

        assert result.entities == {}

    @pytest.mark.asyncio
    async def test_extract_entities_value_error_returns_empty(self):
        """LangchainLLMNER returns empty result on ValueError."""
        from mnemotree.ner.llm import LangchainLLMNER

        mock_llm = MagicMock()
        ner = LangchainLLMNER(llm=mock_llm)

        with patch.object(ner, "parser") as mock_parser:
            mock_parser.get_format_instructions.return_value = "format"

            mock_chain = AsyncMock()
            mock_chain.ainvoke.side_effect = ValueError("value error")

            with patch("mnemotree.ner.llm.PromptTemplate") as mock_prompt:
                mock_prompt.return_value.__or__ = lambda self, other: MagicMock(
                    __or__=lambda self, other: mock_chain
                )

                result = await ner.extract_entities("Some text")

        assert result.entities == {}

    @pytest.mark.asyncio
    async def test_mention_extraction_finds_all_occurrences(self):
        """LangchainLLMNER finds all occurrences of entity in text."""
        from mnemotree.ner.llm import LangchainLLMNER

        mock_llm = MagicMock()
        ner = LangchainLLMNER(llm=mock_llm)

        with patch.object(ner, "parser") as mock_parser:
            mock_parser.get_format_instructions.return_value = "format"

            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = {
                "entities": {"NYC": "LOC"},
                "mentions": {},
                "confidence": {},
            }

            with patch("mnemotree.ner.llm.PromptTemplate") as mock_prompt:
                mock_prompt.return_value.__or__ = lambda self, other: MagicMock(
                    __or__=lambda self, other: mock_chain
                )

                text = "I flew from NYC to LA and then back to NYC"
                result = await ner.extract_entities(text)

        # Should find NYC twice
        assert len(result.mentions.get("NYC", [])) == 2


class TestSparkNLPNER:
    """Tests for SparkNLPNER backend with mocked pyspark."""

    def test_import_error_without_pyspark(self):
        """SparkNLPNER raises ImportError when pyspark not installed."""
        import importlib
        import sys

        with patch.dict(sys.modules, {"pyspark": None}):
            import mnemotree.ner.spark_nlp as spark_module
            importlib.reload(spark_module)

            with pytest.raises(ImportError, match="SparkNLPNER requires"):
                spark_module.SparkNLPNER(spark=MagicMock(), pipeline_model=MagicMock())

    def test_initialization(self):
        """SparkNLPNER stores configuration correctly."""
        import importlib
        import sys

        mock_pyspark = MagicMock()

        with patch.dict(sys.modules, {"pyspark": mock_pyspark}):
            import mnemotree.ner.spark_nlp as spark_module
            importlib.reload(spark_module)

            mock_spark = MagicMock()
            mock_pipeline = MagicMock()

            ner = spark_module.SparkNLPNER(
                spark=mock_spark,
                pipeline_model=mock_pipeline,
                input_col="custom_input",
                output_col="custom_output",
            )

            assert ner._spark is mock_spark
            assert ner._pipeline_model is mock_pipeline
            assert ner._input_col == "custom_input"
            assert ner._output_col == "custom_output"

    def test_item_to_dict_with_asDict(self):
        """_item_to_dict handles objects with asDict method."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        mock_item = MagicMock()
        mock_item.asDict.return_value = {"result": "test", "begin": 0}

        result = SparkNLPNER._item_to_dict(mock_item)

        assert result == {"result": "test", "begin": 0}
        mock_item.asDict.assert_called_once_with(recursive=True)

    def test_item_to_dict_with_dict(self):
        """_item_to_dict handles dict items."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        item = {"result": "test", "begin": 0}
        result = SparkNLPNER._item_to_dict(item)

        assert result == {"result": "test", "begin": 0}

    def test_item_to_dict_with_non_dict(self):
        """_item_to_dict returns None for non-dict items."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        mock_item = MagicMock(spec=[])  # No asDict method

        result = SparkNLPNER._item_to_dict(mock_item)

        assert result is None

    def test_coerce_span_valid(self):
        """_coerce_span converts valid span values."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        start, end = SparkNLPNER._coerce_span(5, 10)

        assert start == 5
        assert end == 11  # end + 1

    def test_coerce_span_with_none(self):
        """_coerce_span handles None values."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        start, end = SparkNLPNER._coerce_span(None, None)

        assert start is None
        assert end is None

    def test_coerce_span_invalid_type(self):
        """_coerce_span returns None for invalid types."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        start, end = SparkNLPNER._coerce_span("invalid", [1, 2])

        assert start is None
        assert end is None

    def test_coerce_score_valid(self):
        """_coerce_score converts valid score."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        score = SparkNLPNER._coerce_score(0.95)
        assert score == pytest.approx(0.95)

    def test_coerce_score_none(self):
        """_coerce_score handles None."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        score = SparkNLPNER._coerce_score(None)
        assert score is None

    def test_coerce_score_invalid(self):
        """_coerce_score returns None for invalid values."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        score = SparkNLPNER._coerce_score("not a number")
        assert score is None

    def test_update_confidence_with_score(self):
        """_update_confidence updates with max score."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        confidence = {"entity1": 0.5}
        SparkNLPNER._update_confidence("entity1", 0.8, confidence)

        assert confidence["entity1"] == pytest.approx(0.8)

    def test_update_confidence_keeps_higher(self):
        """_update_confidence keeps higher existing score."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        confidence = {"entity1": 0.9}
        SparkNLPNER._update_confidence("entity1", 0.5, confidence)

        assert confidence["entity1"] == pytest.approx(0.9)

    def test_update_confidence_with_none_score(self):
        """_update_confidence ignores None scores."""
        from mnemotree.ner.spark_nlp import SparkNLPNER

        confidence = {"entity1": 0.5}
        SparkNLPNER._update_confidence("entity1", None, confidence)

        assert confidence["entity1"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self):
        """SparkNLPNER extracts entities from Spark output."""
        import importlib
        import sys

        mock_pyspark = MagicMock()

        with patch.dict(sys.modules, {"pyspark": mock_pyspark}):
            import mnemotree.ner.spark_nlp as spark_module
            importlib.reload(spark_module)

            mock_spark = MagicMock()
            mock_pipeline = MagicMock()

            # Mock the DataFrame chain
            mock_df = MagicMock()
            mock_spark.createDataFrame.return_value = mock_df

            mock_transformed = MagicMock()
            mock_pipeline.transform.return_value = mock_transformed

            mock_selected = MagicMock()
            mock_transformed.select.return_value = mock_selected

            # Mock the output row
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, i: [
                {
                    "result": "Google",
                    "begin": 0,
                    "end": 5,
                    "metadata": {"entity": "ORG", "confidence": 0.95},
                },
                {
                    "result": "California",
                    "begin": 19,
                    "end": 28,
                    "metadata": {"entity": "LOC", "confidence": 0.88},
                },
            ]
            mock_selected.first.return_value = mock_row

            ner = spark_module.SparkNLPNER(spark=mock_spark, pipeline_model=mock_pipeline)
            result = await ner.extract_entities("Google is based in California")

            assert "Google" in result.entities
            assert result.entities["Google"] == "ORG"
            assert "California" in result.entities
            assert result.entities["California"] == "LOC"
            assert result.confidence["Google"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_extract_entities_empty_output(self):
        """SparkNLPNER handles empty output."""
        import importlib
        import sys

        mock_pyspark = MagicMock()

        with patch.dict(sys.modules, {"pyspark": mock_pyspark}):
            import mnemotree.ner.spark_nlp as spark_module
            importlib.reload(spark_module)

            mock_spark = MagicMock()
            mock_pipeline = MagicMock()

            mock_df = MagicMock()
            mock_spark.createDataFrame.return_value = mock_df
            mock_transformed = MagicMock()
            mock_pipeline.transform.return_value = mock_transformed
            mock_selected = MagicMock()
            mock_transformed.select.return_value = mock_selected
            mock_selected.first.return_value = None  # No results

            ner = spark_module.SparkNLPNER(spark=mock_spark, pipeline_model=mock_pipeline)
            result = await ner.extract_entities("No entities here")

            assert result.entities == {}
            assert result.mentions == {}

    @pytest.mark.asyncio
    async def test_extract_entities_skips_empty_result(self):
        """SparkNLPNER skips items with empty result text."""
        import importlib
        import sys

        mock_pyspark = MagicMock()

        with patch.dict(sys.modules, {"pyspark": mock_pyspark}):
            import mnemotree.ner.spark_nlp as spark_module
            importlib.reload(spark_module)

            mock_spark = MagicMock()
            mock_pipeline = MagicMock()

            mock_df = MagicMock()
            mock_spark.createDataFrame.return_value = mock_df
            mock_transformed = MagicMock()
            mock_pipeline.transform.return_value = mock_transformed
            mock_selected = MagicMock()
            mock_transformed.select.return_value = mock_selected

            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, i: [
                {"result": "  ", "metadata": {"entity": "ORG"}},  # Empty result
                {"result": "Valid", "metadata": {"entity": "ORG"}},
            ]
            mock_selected.first.return_value = mock_row

            ner = spark_module.SparkNLPNER(spark=mock_spark, pipeline_model=mock_pipeline)
            result = await ner.extract_entities("Text")

            assert "Valid" in result.entities
            assert len(result.entities) == 1

    @pytest.mark.asyncio
    async def test_extract_entities_uses_fallback_entity_type(self):
        """SparkNLPNER uses fallback for entity type."""
        import importlib
        import sys

        mock_pyspark = MagicMock()

        with patch.dict(sys.modules, {"pyspark": mock_pyspark}):
            import mnemotree.ner.spark_nlp as spark_module
            importlib.reload(spark_module)

            mock_spark = MagicMock()
            mock_pipeline = MagicMock()

            mock_df = MagicMock()
            mock_spark.createDataFrame.return_value = mock_df
            mock_transformed = MagicMock()
            mock_pipeline.transform.return_value = mock_transformed
            mock_selected = MagicMock()
            mock_transformed.select.return_value = mock_selected

            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, i: [
                {"result": "Test", "metadata": {}},  # No entity type in metadata
            ]
            mock_selected.first.return_value = mock_row

            ner = spark_module.SparkNLPNER(spark=mock_spark, pipeline_model=mock_pipeline)
            result = await ner.extract_entities("Test")

            assert result.entities.get("Test") == "ENTITY"  # Default fallback
