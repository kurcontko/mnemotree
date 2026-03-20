"""Smoke tests for benchmark adapters — no external data or API keys needed."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure benchmarks package is importable
BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from benchmarks.adapters.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkDataset,
    CaseResult,
    ExpectedAnswer,
)
from benchmarks.adapters.locomo import LoCoMoAdapter
from benchmarks.adapters.longmemeval import LongMemEvalAdapter
from benchmarks.adapters.personamem import PersonaMemAdapter
from benchmarks.adapters.prefeval import PrefEvalAdapter
from benchmarks.adapters.halumem import HaluMemAdapter
from benchmarks.adapters.memoryagentbench import MemoryAgentBenchAdapter
from benchmarks.lib.metrics import mean, ndcg_at_k, precision_at_k, recall_at_k, session_recall
from benchmarks.lib.results import build_result


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    LoCoMoAdapter,
    LongMemEvalAdapter,
    PersonaMemAdapter,
    PrefEvalAdapter,
    HaluMemAdapter,
    MemoryAgentBenchAdapter,
])
def test_adapter_is_benchmark_adapter(cls):
    assert isinstance(cls(), BenchmarkAdapter)


# ---------------------------------------------------------------------------
# ADAPTERS registry (all 6 registered)
# ---------------------------------------------------------------------------

def test_all_adapters_registered():
    from benchmarks.run_external import _register_adapters, ADAPTERS
    ADAPTERS.clear()
    _register_adapters()
    expected = {"locomo", "longmemeval", "personamem", "prefeval", "halumem", "memoryagentbench"}
    assert set(ADAPTERS.keys()) == expected


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

def test_locomo_load():
    adapter = LoCoMoAdapter()
    try:
        ds = adapter.load("locomo10")
    except FileNotFoundError:
        pytest.skip("LoCoMo data not found")
    assert ds.name == "locomo"
    assert ds.split == "locomo10"
    assert len(ds.cases) > 0
    assert ds.cases[0].question
    assert ds.cases[0].expected.text


def test_locomo_aggregate():
    adapter = LoCoMoAdapter()
    results = [
        CaseResult(case_id="q1", question="?", predicted="a", expected="a", score=1.0, category="single_hop"),
        CaseResult(case_id="q2", question="?", predicted="b", expected="c", score=0.0, category="temporal"),
        CaseResult(case_id="q3", question="?", predicted="d", expected="d", score=0.5, category="single_hop"),
    ]
    agg = adapter.aggregate(results)
    assert agg["total"] == 3
    assert agg["accuracy"] == pytest.approx(0.5)
    assert agg["by_category"]["single_hop"]["count"] == 2
    assert agg["by_category"]["single_hop"]["accuracy"] == pytest.approx(0.75)
    assert agg["by_category"]["temporal"]["count"] == 1


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

def test_longmemeval_load_missing_data():
    adapter = LongMemEvalAdapter()
    with pytest.raises(FileNotFoundError, match="LongMemEval dataset not found"):
        adapter.load("oracle", data_dir="/tmp/nonexistent_dir_xyz")


def test_longmemeval_load_with_synthetic_data():
    data = [
        {
            "question_id": "test_q1",
            "question": "What happened on Monday?",
            "answer": "Alice went to the park",
            "question_type": "information_extraction",
            "haystack_sessions": [
                {
                    "session_id": "s1",
                    "date": "2024-01-15",
                    "turns": [
                        {"role": "user", "content": "I went to the park on Monday"},
                        {"role": "assistant", "content": "That sounds nice!"},
                    ],
                },
                {
                    "session_id": "s2",
                    "date": "2024-01-20",
                    "turns": [
                        {"role": "user", "content": "Work was busy this week"},
                    ],
                },
            ],
            "answer_session_ids": ["s1"],
        },
        {
            "question_id": "test_q2",
            "question": "Did Alice mention skiing?",
            "answer": "No, Alice did not mention skiing",
            "question_type": "abstention",
            "haystack_sessions": [],
            "answer_session_ids": [],
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "longmemeval_oracle.json"
        with path.open("w") as f:
            json.dump(data, f)

        adapter = LongMemEvalAdapter()
        ds = adapter.load("oracle", data_dir=tmpdir)

        assert ds.name == "longmemeval"
        assert ds.split == "oracle"
        assert len(ds.cases) == 2

        case1 = ds.cases[0]
        assert case1.case_id == "test_q1"
        assert case1.category == "information_extraction"
        assert len(case1.sessions) == 2
        assert case1.sessions[0].session_id == "s1"
        assert len(case1.sessions[0].turns) == 2
        assert case1.metadata["answer_session_ids"] == ["s1"]

        case2 = ds.cases[1]
        assert case2.expected.answer_type == "abstain"


def test_longmemeval_aggregate():
    adapter = LongMemEvalAdapter()
    results = [
        CaseResult(case_id="q1", question="?", predicted="a", expected="a", score=1.0,
                    category="information_extraction", metadata={"session_recall": 1.0}),
        CaseResult(case_id="q2", question="?", predicted="b", expected="c", score=0.0,
                    category="temporal", metadata={"session_recall": 0.5}),
    ]
    agg = adapter.aggregate(results)
    assert agg["total"] == 2
    assert agg["accuracy"] == pytest.approx(0.5)
    assert agg["session_recall"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# PersonaMem
# ---------------------------------------------------------------------------

def test_personamem_load_missing_data():
    adapter = PersonaMemAdapter()
    with pytest.raises(FileNotFoundError, match="PersonaMem context not found"):
        adapter.load("32k", data_dir="/tmp/nonexistent_personamem_xyz")


def test_personamem_load_with_synthetic_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create shared context JSONL
        context_path = Path(tmpdir) / "shared_context_32k.jsonl"
        context_lines = [
            {"role": "user", "content": "I love Italian food"},
            {"role": "assistant", "content": "That's great! Any favorite dishes?"},
            {"role": "user", "content": "I prefer pasta over pizza now"},
        ]
        with context_path.open("w") as f:
            for line in context_lines:
                f.write(json.dumps(line) + "\n")

        # Create benchmark CSV
        csv_path = Path(tmpdir) / "benchmark.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "user_id", "query_type", "question", "answer",
                "end_index_in_shared_context", "preference_distance",
            ])
            writer.writeheader()
            writer.writerow({
                "user_id": "user_1",
                "query_type": "preference_qa",
                "question": "What food does the user prefer?",
                "answer": "Italian food",
                "end_index_in_shared_context": "2",
                "preference_distance": "1",
            })
            writer.writerow({
                "user_id": "user_1",
                "query_type": "preference_update",
                "question": "Does the user prefer pasta or pizza?",
                "answer": "pasta",
                "end_index_in_shared_context": "3",
                "preference_distance": "0",
            })

        adapter = PersonaMemAdapter()
        ds = adapter.load("32k", data_dir=tmpdir)

        assert ds.name == "personamem"
        assert ds.split == "32k"
        assert len(ds.cases) == 2

        case1 = ds.cases[0]
        assert case1.category == "preference_qa"
        assert case1.metadata["user_id"] == "user_1"
        assert case1.metadata["end_index"] == 2
        assert case1.metadata["preference_distance"] == 1
        assert len(case1.sessions) == 1
        assert len(case1.sessions[0].turns) == 2  # sliced to end_index=2

        case2 = ds.cases[1]
        assert case2.category == "preference_update"
        assert len(case2.sessions[0].turns) == 3


def test_personamem_aggregate():
    adapter = PersonaMemAdapter()
    results = [
        CaseResult(case_id="q1", question="?", predicted="a", expected="a", score=1.0,
                    category="preference_qa", metadata={"preference_distance": 1}),
        CaseResult(case_id="q2", question="?", predicted="b", expected="c", score=0.0,
                    category="preference_update", metadata={"preference_distance": 0}),
        CaseResult(case_id="q3", question="?", predicted="d", expected="d", score=1.0,
                    category="preference_qa", metadata={"preference_distance": 1}),
    ]
    agg = adapter.aggregate(results)
    assert agg["total"] == 3
    assert agg["accuracy"] == pytest.approx(2 / 3)
    assert agg["by_query_type"]["preference_qa"]["count"] == 2
    assert agg["by_query_type"]["preference_qa"]["accuracy"] == pytest.approx(1.0)
    assert agg["by_preference_distance"]["1"]["count"] == 2
    assert agg["by_preference_distance"]["0"]["count"] == 1


# ---------------------------------------------------------------------------
# PrefEval
# ---------------------------------------------------------------------------

def test_prefeval_load_missing_data():
    adapter = PrefEvalAdapter()
    with pytest.raises(FileNotFoundError, match="PrefEval dataset not found"):
        adapter.load("explicit", data_dir="/tmp/nonexistent_prefeval_xyz")


def test_prefeval_load_with_synthetic_data():
    data = [
        {
            "preference": "I prefer dark mode in all applications",
            "question": "Should we use light or dark theme?",
            "answer": "dark theme",
            "topic": "ui_preferences",
        },
        {
            "preference": "I am vegetarian",
            "question": "What meal should I order?",
            "answer": "vegetarian option",
            "topic": "food",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prefeval_explicit.json"
        with path.open("w") as f:
            json.dump(data, f)

        adapter = PrefEvalAdapter()
        ds = adapter.load("explicit", data_dir=tmpdir)

        assert ds.name == "prefeval"
        assert ds.split == "explicit"
        assert len(ds.cases) == 2
        assert ds.cases[0].category == "ui_preferences"
        assert ds.cases[0].metadata["preference"] == "I prefer dark mode in all applications"
        assert ds.cases[1].category == "food"


def test_prefeval_aggregate():
    adapter = PrefEvalAdapter()
    results = [
        CaseResult(case_id="p1", question="?", predicted="a", expected="a", score=1.0, category="food"),
        CaseResult(case_id="p2", question="?", predicted="b", expected="c", score=0.0, category="food"),
        CaseResult(case_id="p3", question="?", predicted="d", expected="d", score=1.0, category="travel"),
    ]
    agg = adapter.aggregate(results)
    assert agg["total"] == 3
    assert agg["accuracy"] == pytest.approx(2 / 3)
    assert agg["by_topic"]["food"]["count"] == 2
    assert agg["by_topic"]["food"]["accuracy"] == pytest.approx(0.5)
    assert agg["by_topic"]["travel"]["count"] == 1


# ---------------------------------------------------------------------------
# HaluMem
# ---------------------------------------------------------------------------

def test_halumem_load_missing_data():
    adapter = HaluMemAdapter()
    with pytest.raises(FileNotFoundError, match="HaluMem dataset not found"):
        adapter.load("extract", data_dir="/tmp/nonexistent_halumem_xyz")


def test_halumem_load_with_synthetic_data():
    data = [
        {
            "conversation": [
                {"role": "user", "content": "I visited Paris last summer"},
                {"role": "assistant", "content": "How was the trip?"},
                {"role": "user", "content": "It was great, I saw the Eiffel Tower"},
            ],
            "memory_points": ["visited Paris", "saw Eiffel Tower"],
            "distractors": ["visited London", "saw Big Ben"],
            "question": "What landmark did the user visit?",
            "answer": "Eiffel Tower",
            "question_type": "factual",
        },
        {
            "conversation": [
                {"role": "user", "content": "I started learning piano in January"},
            ],
            "memory_points": ["learning piano since January"],
            "distractors": ["learning guitar since March"],
            "question": "When did the user start learning piano?",
            "answer": "January",
            "question_type": "temporal",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "halumem_extract.json"
        with path.open("w") as f:
            json.dump(data, f)

        adapter = HaluMemAdapter()
        ds = adapter.load("extract", data_dir=tmpdir)

        assert ds.name == "halumem"
        assert ds.split == "extract"
        assert len(ds.cases) == 2

        case1 = ds.cases[0]
        assert case1.category == "factual"
        assert len(case1.sessions) == 1
        assert len(case1.sessions[0].turns) == 3
        assert case1.metadata["memory_points"] == ["visited Paris", "saw Eiffel Tower"]
        assert case1.metadata["distractors"] == ["visited London", "saw Big Ben"]

        case2 = ds.cases[1]
        assert case2.category == "temporal"


def test_halumem_aggregate():
    adapter = HaluMemAdapter()
    results = [
        CaseResult(case_id="h1", question="?", predicted="a", expected="a", score=1.0,
                    category="factual", metadata={"hallucination_detected": False}),
        CaseResult(case_id="h2", question="?", predicted="b", expected="c", score=0.0,
                    category="temporal", metadata={"hallucination_detected": True}),
        CaseResult(case_id="h3", question="?", predicted="d", expected="d", score=1.0,
                    category="factual", metadata={"hallucination_detected": False}),
    ]
    agg = adapter.aggregate(results)
    assert agg["total"] == 3
    assert agg["accuracy"] == pytest.approx(2 / 3)
    assert agg["hallucination_rate"] == pytest.approx(1 / 3)
    assert agg["by_question_type"]["factual"]["count"] == 2
    assert agg["by_question_type"]["factual"]["hallucination_rate"] == pytest.approx(0.0)
    assert agg["by_question_type"]["temporal"]["hallucination_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MemoryAgentBench
# ---------------------------------------------------------------------------

def test_memoryagentbench_load_missing_data():
    adapter = MemoryAgentBenchAdapter()
    with pytest.raises(FileNotFoundError, match="MemoryAgentBench dataset not found"):
        adapter.load("ar", data_dir="/tmp/nonexistent_mab_xyz")


def test_memoryagentbench_load_with_synthetic_data():
    data = [
        {
            "doc_id": "doc_001",
            "chunks": [
                {"chunk_idx": 0, "text": "The capital of France is Paris."},
                {"chunk_idx": 1, "text": "Paris is known for the Eiffel Tower."},
            ],
            "queries": [
                {"query": "What is the capital of France?", "answer": "Paris", "competency": "ar"},
                {"query": "What is Paris known for?", "answer": "Eiffel Tower", "competency": "ar"},
            ],
        },
        {
            "doc_id": "doc_002",
            "chunks": [
                {"chunk_idx": 0, "text": "Python was created by Guido van Rossum."},
            ],
            "queries": [
                {"query": "Who created Python?", "answer": "Guido van Rossum", "competency": "ttl"},
            ],
            "competency": "ttl",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memoryagentbench_ar.json"
        with path.open("w") as f:
            json.dump(data, f)

        adapter = MemoryAgentBenchAdapter()
        ds = adapter.load("ar", data_dir=tmpdir)

        assert ds.name == "memoryagentbench"
        assert ds.split == "ar"
        assert len(ds.cases) == 3

        case1 = ds.cases[0]
        assert case1.metadata["doc_id"] == "doc_001"
        assert case1.metadata["sample_id"] == "doc_001"
        assert case1.metadata["chunk_count"] == 2
        assert case1.category == "ar"
        assert len(case1.sessions) == 1
        assert len(case1.sessions[0].turns) == 2

        case3 = ds.cases[2]
        assert case3.category == "ttl"
        assert case3.metadata["doc_id"] == "doc_002"


def test_memoryagentbench_aggregate():
    adapter = MemoryAgentBenchAdapter()
    results = [
        CaseResult(case_id="m1", question="?", predicted="a", expected="a", score=1.0, category="ar"),
        CaseResult(case_id="m2", question="?", predicted="b", expected="c", score=0.0, category="ar"),
        CaseResult(case_id="m3", question="?", predicted="d", expected="d", score=1.0, category="ttl"),
        CaseResult(case_id="m4", question="?", predicted="e", expected="f", score=0.5, category="lru"),
    ]
    agg = adapter.aggregate(results)
    assert agg["total"] == 4
    assert agg["accuracy"] == pytest.approx(0.625)
    assert agg["by_competency"]["ar"]["count"] == 2
    assert agg["by_competency"]["ar"]["accuracy"] == pytest.approx(0.5)
    assert agg["by_competency"]["ttl"]["count"] == 1
    assert agg["by_competency"]["ttl"]["accuracy"] == pytest.approx(1.0)
    assert agg["by_competency"]["lru"]["count"] == 1


# ---------------------------------------------------------------------------
# Shared metrics
# ---------------------------------------------------------------------------

def test_precision_recall():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "e"}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(3 / 5)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)


def test_session_recall_metric():
    assert session_recall({"s1", "s2"}, {"s1", "s2", "s3"}) == pytest.approx(2 / 3)
    assert session_recall(set(), {"s1"}) == pytest.approx(0.0)
    assert session_recall({"s1"}, set()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def test_build_result():
    result = build_result(
        benchmark="test",
        split="tiny",
        summary={"accuracy": 0.5},
        per_case_results=[
            CaseResult(case_id="q1", question="?", predicted="a", expected="b", score=0.5),
        ],
        config={"k": 10},
    )
    assert result["benchmark"] == "test"
    assert result["split"] == "tiny"
    assert len(result["per_case_results"]) == 1
    assert "timestamp" in result


# ---------------------------------------------------------------------------
# Runner CLI arg parsing
# ---------------------------------------------------------------------------

def test_runner_arg_parsing():
    sys.argv = [
        "run_external.py",
        "--benchmark", "locomo",
        "--split", "locomo10",
        "--limit-cases", "5",
        "--k", "10",
        "--no-ner",
    ]
    from benchmarks.run_external import parse_args
    args = parse_args()
    assert args.benchmark == "locomo"
    assert args.split == "locomo10"
    assert args.limit_cases == 5
    assert args.k == 10
    assert args.no_ner is True
