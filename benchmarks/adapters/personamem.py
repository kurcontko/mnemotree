"""PersonaMem benchmark adapter.

Covers: evolving user preferences, personalization across sessions.
180+ simulated user histories, up to 60 sessions, 15 scenarios, 7 query types.

Splits: 32k, 128k
Source: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2

Expected layout in datasets/PersonaMem/:
  shared_context_32k.jsonl   — one JSON object per line, each a conversation turn
  shared_context_128k.jsonl
  benchmark.csv              — columns: user_id, query_type, question, answer,
                               end_index_in_shared_context, preference_distance, ...
"""

from __future__ import annotations

import csv
import json
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
DEFAULT_DATA_DIR = WORKTREE_ROOT / "benchmarks" / "datasets" / "PersonaMem"

QUERY_TYPES = [
    "preference_qa",
    "preference_update",
    "preference_contradiction",
    "temporal_preference",
    "multi_hop_preference",
    "counterfactual",
    "adversarial",
]


class PersonaMemAdapter:
    """Adapter for PersonaMem (personalized preference memory).

    Each case consists of a user's conversation history (sliced to
    ``end_index_in_shared_context``) plus a question about their preferences.
    """

    name = "personamem"

    def load(self, split: str = "32k", data_dir: str | None = None) -> BenchmarkDataset:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        context_path = base / f"shared_context_{split}.jsonl"
        benchmark_path = base / "benchmark.csv"

        if not context_path.exists():
            raise FileNotFoundError(
                f"PersonaMem context not found: {context_path}. "
                f"Download with: huggingface-cli download bowen-upenn/PersonaMem-v2"
            )
        if not benchmark_path.exists():
            raise FileNotFoundError(f"PersonaMem benchmark CSV not found: {benchmark_path}")

        # Load shared context lines
        context_lines: list[dict[str, Any]] = []
        with context_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    context_lines.append(json.loads(line))

        # Load benchmark questions
        cases: list[BenchmarkCase] = []
        with benchmark_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                end_idx = int(row.get("end_index_in_shared_context", len(context_lines)))
                user_id = row.get("user_id", f"user_{len(cases)}")
                query_type = row.get("query_type", "unknown")
                question = row["question"]
                answer = row.get("answer", "")
                pref_distance = row.get("preference_distance", "")

                # Build conversation history up to end_index
                history = context_lines[:end_idx]
                turns = [
                    {"speaker": turn.get("role", turn.get("speaker", "user")),
                     "text": turn.get("content", turn.get("text", ""))}
                    for turn in history
                ]
                session = ConversationSession(
                    session_id=f"{user_id}_history",
                    turns=turns,
                )

                # Detect answer type
                answer_type = "free_text"
                choices = row.get("choices", "")
                if choices:
                    answer_type = "multiple_choice"

                metadata: dict[str, Any] = {
                    "user_id": user_id,
                    "end_index": end_idx,
                    "query_type": query_type,
                }
                if pref_distance:
                    try:
                        metadata["preference_distance"] = int(pref_distance)
                    except ValueError:
                        metadata["preference_distance"] = pref_distance
                if choices:
                    metadata["choices"] = choices

                cases.append(BenchmarkCase(
                    case_id=f"{user_id}_q{len(cases)}",
                    question=question,
                    expected=ExpectedAnswer(text=answer, answer_type=answer_type),
                    sessions=[session],
                    category=query_type,
                    metadata=metadata,
                ))

        return BenchmarkDataset(name="personamem", split=split, cases=cases)

    async def ingest(self, memory_core: Any, case: BenchmarkCase) -> int:
        count = 0
        for session in case.sessions:
            for turn in session.turns:
                content = f"{turn['speaker']}: {turn['text']}"
                await memory_core.remember(
                    content=content,
                    context={
                        "speaker": turn["speaker"],
                        "session_id": session.session_id,
                        "user_id": case.metadata.get("user_id"),
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
            # For multiple-choice, include choices in the prompt
            q = case.question
            choices = case.metadata.get("choices", "")
            if choices:
                q = f"{q}\nChoices: {choices}"
            predicted = await generate_answer(context, q)

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
            metadata={
                "user_id": case.metadata.get("user_id"),
                "preference_distance": case.metadata.get("preference_distance"),
            },
        )

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        if not results:
            return {"accuracy": 0.0, "total": 0}

        by_query_type: dict[str, list[float]] = {}
        by_distance: dict[str, list[float]] = {}
        for r in results:
            by_query_type.setdefault(r.category or "unknown", []).append(r.score)
            dist = r.metadata.get("preference_distance")
            if dist is not None:
                by_distance.setdefault(str(dist), []).append(r.score)

        return {
            "accuracy": mean([r.score for r in results]),
            "total": len(results),
            "by_query_type": {
                qt: {"accuracy": mean(scores), "count": len(scores)}
                for qt, scores in sorted(by_query_type.items())
            },
            "by_preference_distance": {
                d: {"accuracy": mean(scores), "count": len(scores)}
                for d, scores in sorted(by_distance.items())
            },
        }
