"""HaluMem benchmark adapter.

Tests write-path correctness: extraction, updates, QA hallucination.
Evaluates whether a memory system hallucinates when answering from stored memories.

Splits: extract, update, qa, medium, long
Source: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem

Expected layout in datasets/HaluMem/:
  halumem_{split}.json — list of objects with:
    conversation (list of turns), memory_points (list), distractors (list),
    question, answer, question_type
"""

from __future__ import annotations

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
DEFAULT_DATA_DIR = WORKTREE_ROOT / "benchmarks" / "datasets" / "HaluMem"

QUESTION_TYPES = [
    "factual",
    "temporal",
    "causal",
    "counterfactual",
    "comparative",
    "aggregative",
]


class HaluMemAdapter:
    """Adapter for HaluMem (memory hallucination benchmark).

    Each case contains a multi-turn conversation, ground-truth memory points,
    distractor memory points, and a question. The adapter checks whether the
    system's answer is grounded in real memory points or hallucinates from
    distractors.
    """

    name = "halumem"

    def load(self, split: str = "extract", data_dir: str | None = None) -> BenchmarkDataset:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        data_path = base / f"halumem_{split}.json"

        if not data_path.exists():
            raise FileNotFoundError(
                f"HaluMem dataset not found at {data_path}. "
                f"Download with: huggingface-cli download IAAR-Shanghai/HaluMem"
            )

        with data_path.open() as f:
            data = json.load(f)

        cases: list[BenchmarkCase] = []
        for i, item in enumerate(data):
            question = item.get("question", "")
            answer = item.get("answer", "")
            question_type = item.get("question_type", "unknown")
            memory_points = item.get("memory_points", [])
            distractors = item.get("distractors", [])

            # Build conversation session from turns
            conversation = item.get("conversation", [])
            turns = []
            for turn in conversation:
                turns.append({
                    "speaker": turn.get("role", turn.get("speaker", "user")),
                    "text": turn.get("content", turn.get("text", "")),
                })
            session = ConversationSession(
                session_id=f"halumem_{i}",
                turns=turns,
            )

            cases.append(BenchmarkCase(
                case_id=f"halumem_{split}_{i}",
                question=question,
                expected=ExpectedAnswer(text=answer),
                sessions=[session] if turns else [],
                category=question_type,
                metadata={
                    "memory_points": memory_points,
                    "distractors": distractors,
                    "split": split,
                },
            ))

        return BenchmarkDataset(name="halumem", split=split, cases=cases)

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

        # Check for hallucination: does predicted text reference distractors?
        hallucination_detected = False
        if predicted:
            distractors = case.metadata.get("distractors", [])
            predicted_lower = predicted.lower()
            for distractor in distractors:
                if isinstance(distractor, str) and distractor.lower() in predicted_lower:
                    hallucination_detected = True
                    break

        return CaseResult(
            case_id=case.case_id,
            question=case.question,
            predicted=predicted,
            expected=case.expected.text,
            score=score,
            category=case.category,
            retrieved_ids=retrieved_ids,
            metadata={"hallucination_detected": hallucination_detected},
        )

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        if not results:
            return {"accuracy": 0.0, "total": 0}

        by_type: dict[str, list[CaseResult]] = {}
        for r in results:
            by_type.setdefault(r.category or "unknown", []).append(r)

        hallucinations = sum(
            1 for r in results if r.metadata.get("hallucination_detected", False)
        )

        return {
            "accuracy": mean([r.score for r in results]),
            "hallucination_rate": hallucinations / len(results),
            "total": len(results),
            "by_question_type": {
                qt: {
                    "accuracy": mean([r.score for r in qt_results]),
                    "hallucination_rate": sum(
                        1 for r in qt_results if r.metadata.get("hallucination_detected", False)
                    ) / len(qt_results),
                    "count": len(qt_results),
                }
                for qt, qt_results in sorted(by_type.items())
            },
        }
