"""Tests for MS-RAG query decomposition."""

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.query_decomposition import QueryDecomposer, rrf_merge


class MockResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str) -> MockResponse:
        self.calls.append(prompt)
        return MockResponse(self.response)


def _make_memory(memory_id: str, importance: float = 0.5) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=f"content-{memory_id}",
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        embedding=[0.1] * 3,
    )


class TestQueryDecomposer:
    @pytest.mark.asyncio
    async def test_simple_query_returns_single(self) -> None:
        llm = MockLLM(response="What is Python?")
        decomposer = QueryDecomposer(llm=llm)  # type: ignore[arg-type]

        result = await decomposer.decompose("What is Python?")
        assert result == ["What is Python?"]

    @pytest.mark.asyncio
    async def test_compound_query_returns_multiple(self) -> None:
        llm = MockLLM(response="What is Python?\nWho created Python?\nWhen was it released?")
        decomposer = QueryDecomposer(llm=llm)  # type: ignore[arg-type]

        result = await decomposer.decompose("Tell me about Python")
        assert result == ["What is Python?", "Who created Python?", "When was it released?"]

    @pytest.mark.asyncio
    async def test_respects_max_sub_queries(self) -> None:
        llm = MockLLM(response="Q1\nQ2\nQ3\nQ4\nQ5\nQ6")
        decomposer = QueryDecomposer(llm=llm, max_sub_queries=3)  # type: ignore[arg-type]

        result = await decomposer.decompose("complex question")
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_empty_response_returns_original(self) -> None:
        llm = MockLLM(response="")
        decomposer = QueryDecomposer(llm=llm)  # type: ignore[arg-type]

        result = await decomposer.decompose("original query")
        assert result == ["original query"]

    @pytest.mark.asyncio
    async def test_strips_blank_lines(self) -> None:
        llm = MockLLM(response="Q1\n\nQ2\n  \nQ3")
        decomposer = QueryDecomposer(llm=llm)  # type: ignore[arg-type]

        result = await decomposer.decompose("test")
        assert result == ["Q1", "Q2", "Q3"]

    @pytest.mark.asyncio
    async def test_default_max_sub_queries(self) -> None:
        llm = MockLLM(response="Q1")
        decomposer = QueryDecomposer(llm=llm)  # type: ignore[arg-type]
        assert decomposer.max_sub_queries == 4


class TestRrfMerge:
    def test_single_result_list(self) -> None:
        m1 = _make_memory("m1")
        m2 = _make_memory("m2")
        result = rrf_merge([[m1, m2]], limit=10)
        assert len(result) == 2
        ids = {m.memory_id for m in result}
        assert ids == {"m1", "m2"}

    def test_deduplicates_across_lists(self) -> None:
        m1 = _make_memory("m1")
        m2 = _make_memory("m2")
        m1_dup = _make_memory("m1")
        result = rrf_merge([[m1, m2], [m1_dup]], limit=10)
        ids = [m.memory_id for m in result]
        assert ids.count("m1") == 1

    def test_respects_limit(self) -> None:
        memories = [_make_memory(f"m{i}") for i in range(5)]
        result = rrf_merge([memories], limit=3)
        assert len(result) == 3

    def test_no_limit_returns_all(self) -> None:
        memories = [_make_memory(f"m{i}") for i in range(5)]
        result = rrf_merge([memories], limit=None)
        assert len(result) == 5

    def test_empty_input(self) -> None:
        result = rrf_merge([[]], limit=10)
        assert result == []

    def test_multiple_empty_lists(self) -> None:
        result = rrf_merge([[], []], limit=10)
        assert result == []
