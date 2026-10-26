"""Regression cases for the final embedding/reranking input bound."""

import pytest

from mnemotree.evidence.engine import QUERY_EMBED_TOKENS, model_query
from mnemotree.evidence.packing import TiktokenCounter


@pytest.mark.parametrize(
    "query",
    [
        "😃" * 5900,
        "𠀀" * 5900,
        "𠀀" * 2400 + " ordinary text" * 6000 + "𠀀" * 2400,
        "😀" * 3000 + "a " * 30000 + "😀" * 3000,
    ],
    ids=["short-emoji", "short-rare-cjk", "mixed-cjk", "mixed-emoji"],
)
def test_unicode_model_input_is_bounded_after_endpoint_selection(query):
    counter = TiktokenCounter("o200k_base")
    assert counter.count(query) > QUERY_EMBED_TOKENS
    result = model_query(query, counter=counter)
    assert counter.count(result) <= QUERY_EMBED_TOKENS
    assert result != query
    assert "\n...\n" in result
    assert result.startswith(query[:100])
    assert result.endswith(query[-100:])


@pytest.mark.parametrize("limit", [0, 1, 4, 5, 6, 99, 6000])
def test_counterless_cap_includes_separator(limit):
    result = model_query("a" * 7000, limit=limit)
    assert len(result) <= limit


def test_small_queries_are_preserved_with_configured_counter():
    counter = TiktokenCounter("o200k_base")
    for query in ["hello world", "华沙在哪里？", "😀" * 40, "ordinary words " * 800]:
        assert counter.count(query) <= QUERY_EMBED_TOKENS
        assert model_query(query, counter=counter) == query
