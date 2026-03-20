"""Tests for FactDecomposer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core._internal.fact_decomposer import FactDecomposer


def _make_llm(response_json: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=response_json))
    return llm


class TestIsCompound:
    def test_short_content_not_compound(self):
        decomposer = FactDecomposer(MagicMock(), max_facts=10)
        assert not decomposer.is_compound("Short text.", min_length=200)

    def test_long_with_few_sentences_not_compound(self):
        decomposer = FactDecomposer(MagicMock(), max_facts=10)
        content = "A" * 200 + ". " + "B" * 200 + "."
        # Only 2 sentences → not compound
        assert not decomposer.is_compound(content, min_length=200)

    def test_three_sentences_is_compound(self):
        decomposer = FactDecomposer(MagicMock(), max_facts=10)
        content = "Fact one. Fact two. Fact three."
        assert decomposer.is_compound(content, min_length=20)

    def test_compound_marker_triggers(self):
        decomposer = FactDecomposer(MagicMock(), max_facts=10)
        content = "A" * 200 + " however something else."
        assert decomposer.is_compound(content, min_length=200)

    def test_no_marker_short_sentences_not_compound(self):
        decomposer = FactDecomposer(MagicMock(), max_facts=10)
        content = "A" * 200 + ". " + "B" * 200 + "."
        assert not decomposer.is_compound(content, min_length=200)


class TestDecompose:
    @pytest.mark.asyncio
    async def test_returns_facts_from_llm(self):
        llm = _make_llm('{"facts": ["Fact A.", "Fact B.", "Fact C."]}')
        decomposer = FactDecomposer(llm, max_facts=10)

        facts = await decomposer.decompose("Some compound content.")

        assert facts == ["Fact A.", "Fact B.", "Fact C."]

    @pytest.mark.asyncio
    async def test_respects_max_facts(self):
        facts_json = '{"facts": ["A", "B", "C", "D", "E"]}'
        llm = _make_llm(facts_json)
        decomposer = FactDecomposer(llm, max_facts=3)

        facts = await decomposer.decompose("Some compound content.")

        assert len(facts) == 3
        assert facts == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        decomposer = FactDecomposer(llm, max_facts=10)

        content = "Some content."
        facts = await decomposer.decompose(content)

        assert facts == [content]

    @pytest.mark.asyncio
    async def test_fallback_on_empty_facts(self):
        llm = _make_llm('{"facts": []}')
        decomposer = FactDecomposer(llm, max_facts=10)

        content = "Some content."
        facts = await decomposer.decompose(content)

        assert facts == [content]

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        llm = _make_llm("not valid json")
        decomposer = FactDecomposer(llm, max_facts=10)

        content = "Some content."
        facts = await decomposer.decompose(content)

        assert facts == [content]

    @pytest.mark.asyncio
    async def test_filters_blank_facts(self):
        llm = _make_llm('{"facts": ["Fact A.", "", "  ", "Fact B."]}')
        decomposer = FactDecomposer(llm, max_facts=10)

        facts = await decomposer.decompose("Some compound content.")

        assert facts == ["Fact A.", "Fact B."]
