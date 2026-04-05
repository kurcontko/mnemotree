"""Tests for HyDE (Hypothetical Document Embeddings)."""

import pytest

from mnemotree.core.hyde import HyDEEmbedder


class MockResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    def __init__(self, response: str = "I recall visiting Paris in spring 2024.") -> None:
        self.response = response
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str) -> MockResponse:
        self.calls.append(prompt)
        return MockResponse(self.response)


class MockEmbedder:
    def __init__(self, embedding: list[float] | None = None) -> None:
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.embedded_texts: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return self.embedding


class TestHyDEEmbedder:
    @pytest.mark.asyncio
    async def test_generates_hypothetical_then_embeds(self) -> None:
        llm = MockLLM(response="A hypothetical memory about cooking pasta")
        embedder = MockEmbedder(embedding=[0.5, 0.6, 0.7])
        hyde = HyDEEmbedder(llm=llm, embedder=embedder)  # type: ignore[arg-type]

        result = await hyde.aembed_query("How to cook pasta?")

        assert result == [0.5, 0.6, 0.7]
        assert len(llm.calls) == 1
        assert "How to cook pasta?" in llm.calls[0]
        assert embedder.embedded_texts == ["A hypothetical memory about cooking pasta"]

    @pytest.mark.asyncio
    async def test_embeds_hypothetical_not_original_query(self) -> None:
        llm = MockLLM(response="Generated document")
        embedder = MockEmbedder()
        hyde = HyDEEmbedder(llm=llm, embedder=embedder)  # type: ignore[arg-type]

        await hyde.aembed_query("original query")

        # Should embed the hypothetical, not the original query
        assert embedder.embedded_texts == ["Generated document"]
        assert "original query" not in embedder.embedded_texts

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_hypothetical(self) -> None:
        llm = MockLLM(response="  some memory with spaces  \n")
        embedder = MockEmbedder()
        hyde = HyDEEmbedder(llm=llm, embedder=embedder)  # type: ignore[arg-type]

        await hyde.aembed_query("test query")

        assert embedder.embedded_texts == ["some memory with spaces"]

    @pytest.mark.asyncio
    async def test_handles_non_content_response(self) -> None:
        class StringLLM:
            async def ainvoke(self, prompt: str) -> str:
                return "plain string response"

        embedder = MockEmbedder()
        hyde = HyDEEmbedder(llm=StringLLM(), embedder=embedder)  # type: ignore[arg-type]

        await hyde.aembed_query("test")

        assert embedder.embedded_texts == ["plain string response"]
