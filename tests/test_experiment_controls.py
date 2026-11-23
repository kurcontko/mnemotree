"""Experiment controls: each ablation arm changes one mechanism, and comparisons pair."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemotree import EvidenceEngine, SearchOptions
from mnemotree.cli.evidence import main
from mnemotree.evaluation import (
    distinct_variants,
    evaluate,
    paired_bootstrap,
    retrieval_variants,
)
from mnemotree.service.presets import aml_options


def run_cli():
    # As in test_evidence_cli: keep asyncio.run off the test thread's loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(main).result()


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_aml_variants_alias_the_arms_equal_to_the_profile(tmp_path):
    with EvidenceEngine(tmp_path / "m.db") as memory:
        base = aml_options(memory)
        variants = retrieval_variants(memory, base)
    distinct, aliases = distinct_variants(variants)
    assert base.mode == "lexical"
    for name in ("context_on", "role_hard", "model_query_combined", "token_budget_8000"):
        assert aliases[name] == "baseline"
    assert aliases["mode_lexical"] == "baseline"
    assert set(distinct) | set(aliases) == set(variants)
    assert not set(distinct) & set(aliases)
    # The arms the profile needs, each a real change.
    for name in (
        "context_off",
        "merge_off",
        "source_cap_off",
        "role_and_source_cap_off",
        "deep_pool_off",
        "plain_rag",
    ):
        assert name in distinct
    assert distinct["context_off"].neighbor_messages == 0
    rag = distinct["plain_rag"]
    assert (rag.mode, rag.token_budget, rag.candidate_limit) == (
        base.mode,
        base.token_budget,
        base.candidate_limit,
    )
    assert rag.limit == 100 and rag.chunks_per_source is None and rag.role_first is None
    assert not rag.merge_blocks and rag.neighbor_messages == 0
    assert rag.deep_candidate_limit is None


async def test_rows_identify_history_mechanisms_and_context(tmp_path):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [
                    {"id": "a", "role": "user", "content": "zebra stripes"},
                    {"id": "b", "role": "assistant", "content": "a long zebra answer " * 10},
                ],
            }
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [{"id": "q", "namespace": "u", "query": "zebra", "relevant_source_ids": ["a"]}],
    )
    options = SearchOptions(mode="lexical", role_first="user", role_first_min_assistant_share=0.65)
    with EvidenceEngine(tmp_path / "m.db") as memory:
        report = await evaluate(memory, corpus, queries, options, include_context=True)
    row = report["queries"][0]
    assert row["namespace"] == "u"
    assert row["role_first_applied"] is True and row["assistant_share"] > 0.65
    assert row["candidate_limit"] == 100
    assert row["context"].startswith("user: zebra stripes")
    assert len(row["context_sha256"]) == 64
    assert report["summary"]["role_first_applied_rate"] == 1
    assert len(report["manifest"]["search_sha256"]) == 64


def rows(values, clusters):
    return [
        {"id": f"q{i}", "namespace": cluster, "score": value}
        for i, (value, cluster) in enumerate(zip(values, clusters, strict=True))
    ]


def test_paired_bootstrap_resamples_whole_histories():
    clusters = ["a", "a", "b", "b", "c", "c"]
    left = rows([1, 1, 1, 0, 1, None], clusters)
    right = rows([0, 1, 0, 0, 1, 1], clusters)
    result = paired_bootstrap(left, right, "score", resamples=2000, seed=1)
    assert result["clusters"] == 3 and result["pairs"] == 5 and result["dropped_pairs"] == 1
    assert result["difference"] == pytest.approx(0.4)
    low, high = result["interval"]
    assert low <= result["difference"] <= high
    # Drawing histories, the difference never leaves [0, 1]: no history is worse.
    assert low >= 0 and high <= 1
    assert result["leave_one_out"]["by_cluster"]["c"] == pytest.approx(0.5)
    assert paired_bootstrap(left, right, "score", resamples=2000, seed=1) == result


def test_paired_bootstrap_accepts_a_benchmark_statistic():
    clusters = ["a", "a", "a", "b"]
    left = rows([1, 1, 1, 0], clusters)
    right = rows([0, 0, 0, 0], clusters)

    def per_history(sample):
        by = {}
        for row in sample:
            by.setdefault(row["namespace"], []).append(row["score"])
        return sum(sum(v) / len(v) for v in by.values()) / len(by)

    result = paired_bootstrap(left, right, "score", statistic=per_history, resamples=200)
    # Per question 0.75; averaged per history 0.5.
    assert result["difference"] == pytest.approx(0.5)


def test_paired_bootstrap_rejects_unpaired_rows():
    with pytest.raises(ValueError, match="same query IDs"):
        paired_bootstrap(rows([1], ["a"]), rows([1, 0], ["a", "a"]), "score")
    with pytest.raises(ValueError, match="different namespace"):
        paired_bootstrap(rows([1], ["a"]), rows([1], ["b"]), "score")
    with pytest.raises(ValueError, match="no pair"):
        paired_bootstrap(rows([None], ["a"]), rows([1], ["a"]), "score")


def test_cli_presets_and_compare(tmp_path, monkeypatch, capsys):
    corpus = write_rows(
        tmp_path / "corpus.jsonl",
        [
            {
                "namespace": namespace,
                "request_id": f"r{namespace}",
                "session_id": "s",
                "messages": [
                    {"id": f"{namespace}-a", "role": "user", "content": "zebra stripes"},
                    {"id": f"{namespace}-b", "role": "assistant", "content": "other text"},
                ],
            }
            for namespace in ("u", "v")
        ],
    )
    queries = write_rows(
        tmp_path / "queries.jsonl",
        [
            {"id": n, "namespace": n, "query": "zebra", "relevant_source_ids": [f"{n}-a"]}
            for n in ("u", "v")
        ],
    )
    report_path = tmp_path / "ablation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnemotree-evidence",
            "--db",
            str(tmp_path / "db"),
            "--preset",
            "aml",
            "ablate-retrieval",
            str(corpus),
            str(queries),
            "--output",
            str(report_path),
        ],
    )
    run_cli()
    report = json.loads(report_path.read_text())
    assert report["preset"] == "aml"
    assert report["reports"]["baseline"]["manifest"]["search"]["limit"] == 100
    capsys.readouterr()

    output = tmp_path / "compare.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnemotree-evidence",
            "compare",
            str(report_path),
            str(report_path),
            "--left-arm",
            "baseline",
            "--right-arm",
            "context_on",
            "--metric",
            "packed_source_recall",
            "--resamples",
            "100",
            "--output",
            str(output),
        ],
    )
    run_cli()
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result["difference"] == 0 and result["clusters"] == 2

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"search": {"mode": "lexical"}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnemotree-evidence",
            "--db",
            str(tmp_path / "other"),
            "--config",
            str(config),
            "--preset",
            "aml",
            "evaluate",
            str(corpus),
            str(queries),
            "--output",
            str(tmp_path / "x.json"),
        ],
    )
    with pytest.raises(SystemExit):
        run_cli()
