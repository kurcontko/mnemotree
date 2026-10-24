import json

import pytest
from pydantic import ValidationError

from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.evidence.evaluation import (
    AnswerSpan,
    QueryRecord,
    _answer_span_coverage,
    ablate_retrieval,
    evaluate,
    retrieval_variants,
    score_queries,
)


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_span_coverage_uses_character_union_not_duplicate_lengths():
    gold = [AnswerSpan(source_id="a", start=0, end=10), AnswerSpan(source_id="a", start=5, end=15)]
    assert _answer_span_coverage(gold, [("a", 0, 5), ("a", 2, 8), ("other", 0, 15)]) == 8 / 15
    assert _answer_span_coverage(gold, [("a", 0, 15), ("a", 0, 15)]) == 1
    assert _answer_span_coverage([], []) is None


async def test_retrieving_source_does_not_imply_retrieving_its_answer(tmp_path):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "a", "content": "needleXXanswerYY"}],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [
            {
                "id": "q",
                "namespace": "u",
                "query": "needleXX",
                "answer_spans": [{"source_id": "a", "start": 8, "end": 14}],
            }
        ],
    )
    with Memory(tmp_path / "db", chunk_size=8, chunk_overlap=0) as memory:
        report = await evaluate(memory, corpus, queries, SearchOptions(mode="lexical", limit=1))
    row = report["queries"][0]
    assert row["candidate_source_recall"] == row["packed_source_recall"] == 1
    assert row["candidate_answer_span_coverage"] == row["packed_answer_span_coverage"] == 0
    assert row["candidate_source_recall_at_k"] == row["candidate_reciprocal_rank_at_k"] == 1


async def test_neighbor_expansion_can_add_answer_not_present_in_candidates(tmp_path):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [
                    {"id": "a", "content": "needle"},
                    {"id": "b", "content": "the answer"},
                ],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [
            {
                "id": "q",
                "namespace": "u",
                "query": "needle",
                "answer_spans": [{"source_id": "b", "start": 4, "end": 10}],
            }
        ],
    )
    with Memory(tmp_path / "db") as memory:
        report = await evaluate(
            memory, corpus, queries, SearchOptions(mode="lexical", limit=1, neighbor_messages=1)
        )
    row = report["queries"][0]
    assert row["candidate_answer_span_coverage"] == 0
    assert row["packed_answer_span_coverage"] == row["packing_added_answer_span_fraction"] == 1
    assert row["packing_lost_answer_span_fraction"] == 0


async def test_options_are_wired_and_zero_budget_reports_packing_loss(tmp_path):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "a", "content": "zebra"}],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [
            {
                "id": "q",
                "namespace": "u",
                "query": "unmatched",
                "query_options": ["zebra"],
                "answer_spans": [{"source_id": "a", "start": 0, "end": 5}],
            },
            {"id": "unknown", "namespace": "u", "query": "zebra", "relevant_source_ids": []},
        ],
    )
    with Memory(tmp_path / "db") as memory:
        report = await evaluate(
            memory, corpus, queries, SearchOptions(mode="lexical", token_budget=0)
        )
    row = report["queries"][0]
    assert row["candidate_answer_span_coverage"] == row["packing_lost_answer_span_fraction"] == 1
    assert row["packed_answer_span_coverage"] == 0
    assert report["queries"][1]["packed_all_sources"] is None
    assert report["summary"]["gold_queries"] == 1
    assert report["summary"]["no_gold_queries"] == 1


async def test_lexical_ablation_skips_unsupported_modes_and_validates_settings(tmp_path):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "a", "content": "zebra"}],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [
            {
                "id": "q",
                "namespace": "u",
                "query": "zebra",
                "relevant_source_ids": ["a"],
            }
        ],
    )
    baseline = SearchOptions(mode="lexical")
    with Memory(tmp_path / "db") as memory:
        settings = retrieval_variants(memory, baseline)
        assert "mode_dense" not in settings and "mode_hybrid" not in settings
        assert settings["baseline"] == baseline
        assert settings["role_soft"].role_boost == 0.002
        assert settings["role_soft"].role_first == "user"
        assert settings["role_off"].role_first is None
        assert settings["context_on"].neighbor_messages == 1
        assert {settings[f"token_budget_{n}"].token_budget for n in (1000, 2000, 4000, 8000)} == {
            1000,
            2000,
            4000,
            8000,
        }
        report = await ablate_retrieval(memory, corpus, queries, baseline, candidate_k=1)
    assert len(report["reports"]) == len(settings)
    assert report["reports"]["baseline"]["manifest"]["candidate_k"] == 1


async def test_span_bounds_and_namespace_checked_before_search(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("")
    queries = tmp_path / "queries.jsonl"
    with Memory(tmp_path / "db") as memory:
        await memory.add("u", [Message(id="a", content="zebra")], request_id="r", session_id="s")
        for namespace, end, message in [("other", 5, "absent source"), ("u", 6, "bounds")]:
            write_rows(
                queries,
                [
                    {
                        "id": "q",
                        "namespace": namespace,
                        "query": "zebra",
                        "answer_spans": [{"source_id": "a", "start": 0, "end": end}],
                    }
                ],
            )
            with pytest.raises(ValueError, match=message):
                await score_queries(
                    memory, corpus, queries, SearchOptions(mode="lexical"), {"records": 1}
                )


async def test_unlabeled_only_and_empty_query_sets_are_not_scores(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("")
    queries = tmp_path / "queries.jsonl"
    with Memory(tmp_path / "db") as memory:
        for rows, message in [
            ([], "at least one query"),
            ([{"id": "q", "namespace": "u", "query": "zebra"}], "labeled"),
        ]:
            write_rows(queries, rows)
            with pytest.raises(ValueError, match=message):
                await score_queries(
                    memory, corpus, queries, SearchOptions(mode="lexical"), {"records": 1}
                )
        with pytest.raises(ValueError, match="positive integer"):
            await score_queries(
                memory, corpus, queries, SearchOptions(mode="lexical"), {}, candidate_k=0
            )


def test_query_fields_and_span_order_are_validated():
    with pytest.raises(ValidationError):
        QueryRecord(id="q", namespace="u", query="   ")
    with pytest.raises(ValidationError):
        QueryRecord(id="q", namespace="u", query="x", relevant_source_ids=["a", "a"])
    with pytest.raises(ValidationError):
        AnswerSpan(source_id="a", start=3, end=3)


async def test_merged_neighbors_and_chunk_completion_count_each_gold_character_once(tmp_path):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [
                    {"id": "a", "content": "needle"},
                    {"id": "b", "content": "the answer"},
                    {"id": "c", "content": "needle"},
                ],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [
            {
                "id": "q",
                "namespace": "u",
                "query": "needle",
                "answer_spans": [
                    {"source_id": "b", "start": 0, "end": 10},
                    {"source_id": "b", "start": 4, "end": 10},
                ],
            }
        ],
    )
    with Memory(tmp_path / "db") as memory:
        report = await evaluate(
            memory,
            corpus,
            queries,
            SearchOptions(mode="lexical", limit=2, merge_blocks=True, neighbor_messages=1),
        )
    row = report["queries"][0]
    assert row["candidate_answer_span_coverage"] == 0
    assert row["packed_answer_span_coverage"] == 1
    assert row["packing_added_answer_span_fraction"] == 1
    assert row["packed_context_count"] == 1
    assert row["candidate_count"] == 2

    corpus = write_rows(
        tmp_path / "corpus2.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "a", "content": "needleXXanswerYY"}],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries2.jsonl",
        [
            {
                "id": "q",
                "namespace": "u",
                "query": "needleXX",
                "answer_spans": [{"source_id": "a", "start": 8, "end": 14}],
            }
        ],
    )
    with Memory(tmp_path / "db2", chunk_size=8, chunk_overlap=0) as memory:
        report = await evaluate(
            memory,
            corpus,
            queries,
            SearchOptions(mode="lexical", limit=1, chunks_per_source=1, merge_blocks=True),
        )
    row = report["queries"][0]
    assert row["candidate_answer_span_coverage"] == 0
    assert row["packed_answer_span_coverage"] == 1
    assert row["packing_added_answer_span_fraction"] == 1
