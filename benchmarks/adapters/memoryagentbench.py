"""MemoryAgentBench adapter.

Evaluates incremental agent memory across four competencies:
accurate retrieval (AR), test-time learning (TTL),
long-range understanding (LRU), conflict resolution (CR).

Splits: ar, ttl, lru, cr
Source: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench

Actual HuggingFace layout (Parquet files):
  data/Accurate_Retrieval-00000-of-00001.parquet
  data/Conflict_Resolution-00000-of-00001.parquet
  data/Test_Time_Learning-00000-of-00001.parquet
  data/Long_Range_Understanding-00000-of-00001.parquet

Each row has:
  context: str       — full text blob (numbered facts or documents)
  questions: list    — array of question strings
  answers: list      — array of answer arrays (multiple valid answers per question)
  metadata: dict     — qa_pair_ids, source, etc.

Also supports pre-converted JSON format (memoryagentbench_{split}.json)
for tests and synthetic data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseResult,
    ConversationSession,
    ExpectedAnswer,
)
from benchmarks.lib.metrics import mean

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = WORKTREE_ROOT / "benchmarks" / "datasets" / "MemoryAgentBench"

COMPETENCIES = ["ar", "ttl", "lru", "cr"]

SPLIT_TO_PARQUET = {
    "ar": "Accurate_Retrieval",
    "ttl": "Test_Time_Learning",
    "lru": "Long_Range_Understanding",
    "cr": "Conflict_Resolution",
}

# Max characters per chunk when splitting context for ingestion
CHUNK_SIZE = 512


def _chunk_context(context: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split context text into chunks, preferring line boundaries."""
    lines = context.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > chunk_size and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


class MemoryAgentBenchAdapter:
    """Adapter for MemoryAgentBench (incremental agent memory benchmark).

    Loads from Parquet (real HuggingFace data) or JSON (synthetic/test data).
    The runner's ``ingested_samples`` set handles inject-once deduplication.
    """

    name = "memoryagentbench"

    def load(self, split: str = "ar", data_dir: str | None = None) -> BenchmarkDataset:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        # Try JSON first (synthetic/test data)
        json_path = base / f"memoryagentbench_{split}.json"
        if json_path.exists():
            return self._load_json(json_path, split)

        # Try Parquet (real HuggingFace data)
        parquet_name = SPLIT_TO_PARQUET.get(split, split)
        parquet_path = base / "data" / f"{parquet_name}-00000-of-00001.parquet"
        if parquet_path.exists():
            return self._load_parquet(parquet_path, split)

        raise FileNotFoundError(
            f"MemoryAgentBench dataset not found for split '{split}' in {base}. "
            f"Tried: {json_path} and {parquet_path}. "
            f"Download with: huggingface-cli download ai-hyz/MemoryAgentBench --repo-type dataset"
        )

    def _load_parquet(self, path: Path, split: str) -> BenchmarkDataset:
        import pandas as pd

        df = pd.read_parquet(path)
        cases: list[BenchmarkCase] = []

        for row_idx, row in df.iterrows():
            context_text = row["context"]
            questions = row["questions"]
            answers = row["answers"]
            metadata = row.get("metadata", {}) or {}
            source = metadata.get("source", f"row_{row_idx}")
            if isinstance(source, (list,)):
                source = source[0] if source else f"row_{row_idx}"
            doc_id = f"{split}_{source}_{row_idx}"

            # Chunk the context for ingestion
            chunks = _chunk_context(context_text)
            turns = [{"speaker": "document", "text": chunk} for chunk in chunks]
            session = ConversationSession(session_id=doc_id, turns=turns)

            qa_pair_ids = metadata.get("qa_pair_ids")

            for qi in range(len(questions)):
                question = questions[qi]
                # answers[qi] is an array of acceptable answers
                answer_arr = answers[qi]
                if hasattr(answer_arr, "tolist"):
                    answer_arr = answer_arr.tolist()
                if isinstance(answer_arr, list):
                    primary = answer_arr[0] if answer_arr else ""
                    alternatives = answer_arr[1:] if len(answer_arr) > 1 else []
                else:
                    primary = str(answer_arr)
                    alternatives = []

                qa_id = ""
                if qa_pair_ids is not None and qi < len(qa_pair_ids):
                    qa_id = str(qa_pair_ids[qi])

                cases.append(BenchmarkCase(
                    case_id=qa_id or f"{doc_id}_q{qi}",
                    question=question,
                    expected=ExpectedAnswer(text=primary, alternatives=alternatives),
                    sessions=[session],
                    category=split,
                    metadata={
                        "sample_id": doc_id,
                        "doc_id": doc_id,
                        "chunk_count": len(chunks),
                        "competency": split,
                        "source": str(source),
                    },
                ))

        return BenchmarkDataset(name="memoryagentbench", split=split, cases=cases)

    def _load_json(self, path: Path, split: str) -> BenchmarkDataset:
        with path.open() as f:
            data = json.load(f)

        cases: list[BenchmarkCase] = []
        for item in data:
            doc_id = item.get("doc_id", f"doc_{len(cases)}")
            chunks = item.get("chunks", [])
            queries = item.get("queries", [])
            competency = item.get("competency", split)

            turns = [{"speaker": "document", "text": c.get("text", "")} for c in chunks]
            session = ConversationSession(session_id=doc_id, turns=turns)

            for qi, query in enumerate(queries):
                question = query.get("query", query.get("question", ""))
                answer = query.get("answer", "")
                q_competency = query.get("competency", competency)

                cases.append(BenchmarkCase(
                    case_id=f"{doc_id}_q{qi}",
                    question=question,
                    expected=ExpectedAnswer(text=answer),
                    sessions=[session],
                    category=q_competency,
                    metadata={
                        "sample_id": doc_id,
                        "doc_id": doc_id,
                        "chunk_count": len(chunks),
                        "competency": q_competency,
                    },
                ))

        return BenchmarkDataset(name="memoryagentbench", split=split, cases=cases)

    async def ingest(self, memory_core: Any, case: BenchmarkCase) -> int:
        count = 0
        doc_id = case.metadata.get("doc_id", "")
        for session in case.sessions:
            for chunk_idx, turn in enumerate(session.turns):
                await memory_core.remember(
                    content=turn["text"],
                    context={
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "session_id": session.session_id,
                    },
                    analyze=False,
                    summarize=False,
                )
                count += 1
        return count

    async def run_case(self, memory_core: Any, case: BenchmarkCase, **kwargs: Any) -> CaseResult:
        k = kwargs.get("k", 20)
        generate_answer = kwargs.get("generate_answer")
        judge_answer = kwargs.get("judge_answer")

        retrieved = await memory_core.recall(case.question, limit=k, scoring=True, update_access=False)
        retrieved_ids = [getattr(m, "memory_id", "") for m in retrieved]
        context = "\n".join(f"[{i+1}] {m.content}" for i, m in enumerate(retrieved))

        predicted = ""
        if generate_answer:
            predicted = await generate_answer(context, case.question)

        score = 0.0
        if judge_answer and predicted:
            score = await judge_answer(case.question, case.expected.text, predicted)

        return CaseResult(
            case_id=case.case_id,
            question=case.question,
            predicted=predicted,
            expected=case.expected.text,
            score=score,
            category=case.category,
            retrieved_ids=retrieved_ids,
        )

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        if not results:
            return {"accuracy": 0.0, "total": 0}

        by_competency: dict[str, list[float]] = {}
        for r in results:
            by_competency.setdefault(r.category or "unknown", []).append(r.score)

        return {
            "accuracy": mean([r.score for r in results]),
            "total": len(results),
            "by_competency": {
                comp: {"accuracy": mean(scores), "count": len(scores)}
                for comp, scores in sorted(by_competency.items())
            },
        }
