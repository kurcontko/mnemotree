"""PrefEval benchmark adapter.

Evaluates whether a memory system can recall and apply user preferences.

Splits: explicit, implicit, conflict
Source: https://github.com/amazon-science/PrefEval

Expected layout in datasets/PrefEval/:
  prefeval_{split}.json — list of objects with:
    preference, question, answer, topic, preference_type, ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseResult,
    ExpectedAnswer,
)
from benchmarks.lib.metrics import mean

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = WORKTREE_ROOT / "benchmarks" / "datasets" / "PrefEval"


class PrefEvalAdapter:
    """Adapter for PrefEval (preference-following benchmark).

    Each case has a preference statement (to ingest) and a question
    whose answer depends on remembering that preference.
    """

    name = "prefeval"

    def load(self, split: str = "explicit", data_dir: str | None = None) -> BenchmarkDataset:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        data_path = base / f"prefeval_{split}.json"

        if not data_path.exists():
            raise FileNotFoundError(
                f"PrefEval dataset not found at {data_path}. "
                f"Clone from: git clone https://github.com/amazon-science/PrefEval"
            )

        with data_path.open() as f:
            data = json.load(f)

        cases: list[BenchmarkCase] = []
        for i, item in enumerate(data):
            preference = item.get("preference", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            topic = item.get("topic", "general")
            pref_type = item.get("preference_type", split)

            answer_type = "free_text"
            choices = item.get("choices")
            if choices:
                answer_type = "multiple_choice"

            metadata: dict[str, Any] = {
                "preference": preference,
                "topic": topic,
                "preference_type": pref_type,
            }
            if choices:
                metadata["choices"] = choices

            cases.append(BenchmarkCase(
                case_id=f"pref_{split}_{i}",
                question=question,
                expected=ExpectedAnswer(text=answer, answer_type=answer_type),
                category=topic,
                metadata=metadata,
            ))

        return BenchmarkDataset(name="prefeval", split=split, cases=cases)

    async def ingest(self, memory_core: Any, case: BenchmarkCase) -> int:
        preference = case.metadata.get("preference", "")
        if not preference:
            return 0
        await memory_core.remember(
            content=preference,
            context={
                "topic": case.metadata.get("topic"),
                "type": "preference",
            },
            analyze=False,
            summarize=False,
        )
        return 1

    async def run_case(self, memory_core: Any, case: BenchmarkCase, **kwargs: Any) -> CaseResult:
        k = kwargs.get("k", 20)
        generate_answer = kwargs.get("generate_answer")
        judge_answer = kwargs.get("judge_answer")

        retrieved = await memory_core.recall(case.question, limit=k, scoring=True, update_access=False)
        retrieved_ids = [getattr(m, "memory_id", "") for m in retrieved]
        context = "\n".join(f"[{i+1}] {m.content}" for i, m in enumerate(retrieved))

        predicted = ""
        if generate_answer:
            q = case.question
            choices = case.metadata.get("choices")
            if choices:
                if isinstance(choices, list):
                    choices_str = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(choices))
                else:
                    choices_str = str(choices)
                q = f"{q}\nChoices:\n{choices_str}"
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
        )

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        if not results:
            return {"accuracy": 0.0, "total": 0}

        by_topic: dict[str, list[float]] = {}
        for r in results:
            by_topic.setdefault(r.category or "unknown", []).append(r.score)

        return {
            "accuracy": mean([r.score for r in results]),
            "total": len(results),
            "by_topic": {
                topic: {"accuracy": mean(scores), "count": len(scores)}
                for topic, scores in sorted(by_topic.items())
            },
        }
