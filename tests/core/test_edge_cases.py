"""Tests for edge cases and hardened error handling."""

from __future__ import annotations

import pytest

from mnemotree.core._internal.indexing import BM25Index
from mnemotree.core.query_decomposition import QueryDecomposer


class TestBM25EdgeCases:
    def test_search_after_all_docs_removed(self):
        """Removing all docs should not cause division by zero."""
        index = BM25Index()
        index.add("doc1", ["hello", "world"])
        index.remove("doc1")
        # avgdl is now 0, search should return empty not crash
        results = index.search(["hello"], top_k=5)
        assert results == []

    def test_search_empty_index(self):
        index = BM25Index()
        results = index.search(["hello"], top_k=5)
        assert results == []

    def test_search_with_empty_query(self):
        index = BM25Index()
        index.add("doc1", ["hello", "world"])
        results = index.search([], top_k=5)
        assert results == []


class TestQueryDecomposerEdgeCases:
    @pytest.mark.asyncio
    async def test_none_content_response(self):
        """LLM returning None content should not crash."""

        class _FakeLLM:
            async def ainvoke(self, prompt):
                class _Resp:
                    content = None

                return _Resp()

        decomposer = QueryDecomposer(llm=_FakeLLM())
        result = await decomposer.decompose("What is X?")
        # Should fall back to original query
        assert result == ["What is X?"]

    @pytest.mark.asyncio
    async def test_empty_content_response(self):
        """LLM returning empty string should fall back to original query."""

        class _FakeLLM:
            async def ainvoke(self, prompt):
                class _Resp:
                    content = ""

                return _Resp()

        decomposer = QueryDecomposer(llm=_FakeLLM())
        result = await decomposer.decompose("What is X?")
        assert result == ["What is X?"]


class TestSafeJsonLoads:
    def test_valid_json(self):
        from mnemotree.store.sqlite_vec_store import _safe_json_loads

        assert _safe_json_loads('{"key": "value"}') == {"key": "value"}

    def test_none_input(self):
        from mnemotree.store.sqlite_vec_store import _safe_json_loads

        assert _safe_json_loads(None) == {}

    def test_empty_string(self):
        from mnemotree.store.sqlite_vec_store import _safe_json_loads

        assert _safe_json_loads("") == {}

    def test_malformed_json(self):
        from mnemotree.store.sqlite_vec_store import _safe_json_loads

        assert _safe_json_loads("{invalid json}") == {}

    def test_non_dict_json(self):
        from mnemotree.store.sqlite_vec_store import _safe_json_loads

        assert _safe_json_loads("[1, 2, 3]") == {}
