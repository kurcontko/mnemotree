import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemotree.cli.evidence import main


def run_cli():
    # main() ends in asyncio.run, which clears its thread's event loop when done. On
    # the test thread that would drop the loop pytest-asyncio set there, unclosed.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(main).result()


def test_retrieval_ablation_cli_writes_span_metrics_and_reproducibility(
    tmp_path, monkeypatch, capsys
):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    config = tmp_path / "config.json"
    output = tmp_path / "reports" / "report.json"
    corpus.write_text(
        json.dumps(
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "a", "content": "needleXXanswerYY"}],
            }
        )
        + "\n"
    )
    queries.write_text(
        json.dumps(
            {
                "id": "q",
                "namespace": "u",
                "query": "needleXX",
                "answer_spans": [{"source_id": "a", "start": 8, "end": 14}],
            }
        )
        + "\n"
    )
    config.write_text(
        json.dumps(
            {
                "chunk_size": 8,
                "chunk_overlap": 0,
                "search": {"mode": "lexical"},
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnemotree-evidence",
            "--db",
            str(tmp_path / "db"),
            "--config",
            str(config),
            "ablate-retrieval",
            str(corpus),
            str(queries),
            "--candidate-k",
            "1",
            "--output",
            str(output),
        ],
    )
    run_cli()
    report = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == report["summary"]
    baseline = report["reports"]["baseline"]
    assert baseline["summary"]["packed_source_recall"] == 1
    assert baseline["summary"]["packed_answer_span_coverage"] == 0
    assert baseline["manifest"]["candidate_k"] == 1
    assert len(report["code_identity"]["source_digest"]) == 64
    assert {"role_off", "role_hard", "role_soft", "token_budget_2000"} <= report["reports"].keys()
    assert "mode_dense" not in report["reports"]


def test_cli_search_routes_explicit_options(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "namespace": "u",
                "request_id": "r",
                "session_id": "s",
                "messages": [{"id": "a", "content": "zebra"}],
            }
        )
        + "\n"
    )
    common = ["mnemotree-evidence", "--db", str(tmp_path / "db")]
    monkeypatch.setattr(sys, "argv", [*common, "add", str(corpus)])
    run_cli()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", [*common, "search", "u", "unmatched", "--option", "zebra"])
    run_cli()
    result = json.loads(capsys.readouterr().out)
    assert result["trace"]["channel_counts"]["lexical"] == 1
    assert result["evidence"][0]["source_id"] == "a"


def test_cli_rejects_invalid_candidate_k_before_creating_database(tmp_path, monkeypatch):
    path = tmp_path / "db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnemotree-evidence",
            "--db",
            str(path),
            "ablate-retrieval",
            "corpus",
            "queries",
            "--candidate-k",
            "0",
            "--output",
            "report",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_cli()
    assert exc.value.code == 2
    assert not path.exists()
