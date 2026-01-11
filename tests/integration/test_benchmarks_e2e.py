import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_benchmark_e2e_smoke(tmp_path: Path) -> None:
    """Runs benchmarks/evaluate.py end-to-end on a tiny dataset.

    This is a smoke/integration test to catch pipeline regressions:
    - argument parsing
    - store construction
    - seeding memories
    - recall + metrics aggregation

    It uses an in-memory store and deterministic dummy embeddings, so it should be fast
    and not require external services.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    memories_path = data_dir / "memories.jsonl"
    queries_path = data_dir / "test_queries.jsonl"

    memories = [
        {"content": "Alice likes pizza.", "memory_type": "semantic", "tags": ["alice", "pizza"]},
        {"content": "Bob prefers sushi.", "memory_type": "semantic", "tags": ["bob", "sushi"]},
        {"content": "Carol loves pasta.", "memory_type": "semantic", "tags": ["carol", "pasta"]},
    ]
    with memories_path.open("w") as handle:
        for item in memories:
            handle.write(json.dumps(item) + "\n")

    queries = [
        {
            "query": "What does Alice like?",
            "expected_memories": ["Alice likes pizza."],
            "description": "Simple preference query",
        },
        {
            "query": "Who prefers sushi?",
            "expected_memories": ["Bob prefers sushi."],
            "description": "Simple preference query",
        },
    ]
    with queries_path.open("w") as handle:
        for item in queries:
            handle.write(json.dumps(item) + "\n")

    output_path = tmp_path / "out.json"

    cmd = [
        sys.executable,
        "benchmarks/evaluate.py",
        "--data-dir",
        str(data_dir),
        "--memories-file",
        "memories.jsonl",
        "--queries-file",
        "test_queries.jsonl",
        "--k-values",
        "1,3",
        "--store",
        "inmemory",
        "--mode",
        "lite",
        "--dummy-embeddings",
        "--retrieval-mode",
        "rrf",
        "--enable-bm25",
        "--disable-ner",
        "--disable-keywords",
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    data = json.loads(output_path.read_text())
    assert "summary" in data
    assert data["summary"]["num_memories"] == 3
    assert data["summary"]["num_queries"] == 2

    metrics = data["summary"]["metrics"]
    assert float(metrics["precision@k"]["1"]) >= 0.5
    assert float(metrics["recall@k"]["3"]) >= 0.5
