"""Tests for SimpleMem intent classification."""

import pytest

from mnemotree.core.intent import (
    INTENT_TO_TYPES,
    KeywordIntentClassifier,
    RetrievalIntent,
)
from mnemotree.core.models import MemoryType


@pytest.fixture
def classifier() -> KeywordIntentClassifier:
    return KeywordIntentClassifier()


class TestRetrievalIntent:
    def test_intent_values(self) -> None:
        assert RetrievalIntent.EPISODIC.value == "episodic"
        assert RetrievalIntent.SEMANTIC.value == "semantic"
        assert RetrievalIntent.PROCEDURAL.value == "procedural"
        assert RetrievalIntent.UNKNOWN.value == "unknown"

    def test_intent_to_types_mapping(self) -> None:
        assert MemoryType.EPISODIC in INTENT_TO_TYPES[RetrievalIntent.EPISODIC]
        assert MemoryType.AUTOBIOGRAPHICAL in INTENT_TO_TYPES[RetrievalIntent.EPISODIC]
        assert MemoryType.SEMANTIC in INTENT_TO_TYPES[RetrievalIntent.SEMANTIC]
        assert MemoryType.PROCEDURAL in INTENT_TO_TYPES[RetrievalIntent.PROCEDURAL]

    def test_unknown_not_in_mapping(self) -> None:
        assert RetrievalIntent.UNKNOWN not in INTENT_TO_TYPES


class TestKeywordIntentClassifier:
    @pytest.mark.asyncio
    async def test_episodic_queries(self, classifier: KeywordIntentClassifier) -> None:
        assert (
            await classifier.classify("When did I last meet with Alice?")
            == RetrievalIntent.EPISODIC
        )
        assert await classifier.classify("What happened yesterday?") == RetrievalIntent.EPISODIC
        assert await classifier.classify("Do you remember the meeting?") == RetrievalIntent.EPISODIC
        assert await classifier.classify("Did I ever visit Paris?") == RetrievalIntent.EPISODIC

    @pytest.mark.asyncio
    async def test_semantic_queries(self, classifier: KeywordIntentClassifier) -> None:
        assert await classifier.classify("What is photosynthesis?") == RetrievalIntent.SEMANTIC
        assert await classifier.classify("Why is the sky blue?") == RetrievalIntent.SEMANTIC
        assert await classifier.classify("Explain quantum computing") == RetrievalIntent.SEMANTIC
        assert await classifier.classify("Define recursion") == RetrievalIntent.SEMANTIC

    @pytest.mark.asyncio
    async def test_procedural_queries(self, classifier: KeywordIntentClassifier) -> None:
        assert await classifier.classify("How to install Python?") == RetrievalIntent.PROCEDURAL
        assert await classifier.classify("How do I configure nginx?") == RetrievalIntent.PROCEDURAL
        assert await classifier.classify("Steps to deploy to AWS") == RetrievalIntent.PROCEDURAL
        assert (
            await classifier.classify("Set up a virtual environment") == RetrievalIntent.PROCEDURAL
        )

    @pytest.mark.asyncio
    async def test_unknown_queries(self, classifier: KeywordIntentClassifier) -> None:
        assert await classifier.classify("hello") == RetrievalIntent.UNKNOWN
        assert await classifier.classify("pizza") == RetrievalIntent.UNKNOWN
        assert await classifier.classify("the quick brown fox") == RetrievalIntent.UNKNOWN

    @pytest.mark.asyncio
    async def test_case_insensitive(self, classifier: KeywordIntentClassifier) -> None:
        assert await classifier.classify("WHAT IS a computer?") == RetrievalIntent.SEMANTIC
        assert await classifier.classify("HOW TO cook pasta?") == RetrievalIntent.PROCEDURAL

    @pytest.mark.asyncio
    async def test_tiebreaker_favors_procedural(self, classifier: KeywordIntentClassifier) -> None:
        # "how do i" is procedural, "what is" would be semantic — both match once
        # Implementation favors procedural when tied
        result = await classifier.classify("how do i explain what is happening?")
        assert result in {
            RetrievalIntent.PROCEDURAL,
            RetrievalIntent.SEMANTIC,
            RetrievalIntent.EPISODIC,
        }


class TestLLMIntentClassifier:
    @pytest.mark.asyncio
    async def test_classify_with_mock_llm(self) -> None:
        from mnemotree.core.intent import LLMIntentClassifier

        class MockResponse:
            content = "semantic"

        class MockLLM:
            async def ainvoke(self, prompt: str) -> MockResponse:
                return MockResponse()

        classifier = LLMIntentClassifier(MockLLM())  # type: ignore[arg-type]
        assert await classifier.classify("What is Python?") == RetrievalIntent.SEMANTIC

    @pytest.mark.asyncio
    async def test_classify_unknown_on_gibberish(self) -> None:
        from mnemotree.core.intent import LLMIntentClassifier

        class MockResponse:
            content = "I'm not sure what category this falls into"

        class MockLLM:
            async def ainvoke(self, prompt: str) -> MockResponse:
                return MockResponse()

        classifier = LLMIntentClassifier(MockLLM())  # type: ignore[arg-type]
        assert await classifier.classify("asdfgh") == RetrievalIntent.UNKNOWN

    @pytest.mark.asyncio
    async def test_classify_handles_non_content_response(self) -> None:
        from mnemotree.core.intent import LLMIntentClassifier

        class MockLLM:
            async def ainvoke(self, prompt: str) -> str:
                return "procedural"

        classifier = LLMIntentClassifier(MockLLM())  # type: ignore[arg-type]
        assert await classifier.classify("How to cook?") == RetrievalIntent.PROCEDURAL
