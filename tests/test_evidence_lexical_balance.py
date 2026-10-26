"""Bounded lexical queries retain the question and every explicit option."""

import sqlite3

from mnemotree.evidence.lexical import index_text, match_expression, scoped_match


def terms(query: str, **kwargs: object) -> list[str]:
    expression = match_expression(query, **kwargs)  # type: ignore[arg-type]
    return [] if expression is None else [term.strip('"') for term in expression.split(" OR ")]


def test_long_query_retains_distinctive_tail_and_head() -> None:
    query = " ".join(f"prefix{i}" for i in range(140)) + " distinctiveending"
    selected = terms(query)
    assert len(selected) == 64
    assert selected[0] == "prefix0"
    assert selected[-1] == "distinctiveending"
    assert "prefix70" not in selected
    assert terms(query) == selected


def test_explicit_options_share_slots_after_long_question() -> None:
    query = " ".join(f"question{i}" for i in range(100))
    options = [" ".join(f"choice{option}term{i}" for i in range(40)) for option in range(6)]
    selected = terms(query, options=options)
    assert len(selected) == len(set(selected)) == 64
    assert sum(term.startswith("question") for term in selected) == 32
    counts = [
        sum(term.startswith(f"choice{option}term") for term in selected) for option in range(6)
    ]
    assert max(counts) - min(counts) <= 1
    assert min(counts) >= 5
    for option in range(6):
        assert f"choice{option}term0" in selected
        assert f"choice{option}term39" in selected


def test_unused_option_quota_returns_to_question() -> None:
    query = " ".join(f"question{i}" for i in range(100))
    selected = terms(query, options=["the and", "", "question0 question99 soleoption"])
    assert len(selected) == 64
    assert "soleoption" in selected
    assert sum(term.startswith("question") for term in selected) == 63
    assert "the" not in selected


def test_short_queries_preserve_existing_order_and_fallback() -> None:
    assert match_expression("The Zebra and APPLE zebra 上海") == '"zebra" OR "apple" OR "上海"'
    assert match_expression("the and THE") == '"the" OR "and"'
    assert match_expression("") is None
    assert match_expression("zebra", max_terms=0) is None
    assert terms("zebra", options=["APPLE zebra", "the and", "apple pear"]) == [
        "zebra",
        "apple",
        "pear",
    ]
    assert terms("the", options=["and the"]) == ["the", "and"]


def test_cjk_option_tail_is_searchable_and_namespace_stays_scoped() -> None:
    options = [" ".join(f"prefix{i}" for i in range(90)) + " 我搬到了华沙"]
    expression = scoped_match("tenant", "where moved", options=options)
    assert expression is not None
    assert '"华沙"' in expression
    connection = sqlite3.connect(":memory:")
    connection.execute(
        'CREATE VIRTUAL TABLE evidence USING fts5(scope, text, tokenize="porter unicode61")'
    )
    from mnemotree.evidence.lexical import scope_token

    connection.executemany(
        "INSERT INTO evidence VALUES (?, ?)",
        [
            (scope_token("tenant"), index_text("华沙")),
            (scope_token("other"), index_text("华沙")),
        ],
    )
    assert connection.execute(
        "SELECT count(*) FROM evidence WHERE evidence MATCH ?", (expression,)
    ).fetchone() == (1,)
    connection.close()


def test_hard_maximum_even_when_caller_requests_more() -> None:
    assert len(terms(" ".join(f"term{i}" for i in range(100)), max_terms=200)) == 64
