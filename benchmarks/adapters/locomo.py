"""LoCoMo benchmark adapter — wraps existing EasyLocomo logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseResult,
    ConversationSession,
    EvidenceRef,
    ExpectedAnswer,
)

# Workspace root: mnemotree-multi-bench is a sibling of EasyLocomo
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "EasyLocomo" / "data"


class LoCoMoAdapter:
    """Adapter for the LoCoMo (Long-term Conversational Memory) benchmark."""

    name = "locomo"

    CATEGORY_NAMES = {
        1: "single_hop",
        2: "temporal",
        3: "multi_hop",
        4: "open_domain",
        5: "adversarial",
    }

    def load(self, split: str = "locomo10", data_dir: str | None = None) -> BenchmarkDataset:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        evidence_path = base / f"{split}_evidence.json"
        qa_path = base / f"{split}_qa.json"

        if not evidence_path.exists():
            raise FileNotFoundError(f"LoCoMo evidence not found: {evidence_path}")
        if not qa_path.exists():
            raise FileNotFoundError(f"LoCoMo QA not found: {qa_path}")

        with evidence_path.open() as f:
            evidence_data = json.load(f)
        with qa_path.open() as f:
            qa_data = json.load(f)

        evidence_by_sample = {item["sample_id"]: item for item in evidence_data}
        cases: list[BenchmarkCase] = []

        for sample in qa_data:
            sample_id = sample["sample_id"]
            ev = evidence_by_sample.get(sample_id, {})

            sessions = []
            conv = ev.get("conversation", {})
            for key in sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time")):
                date = conv.get(f"{key}_date_time")
                turns = [{"speaker": t["speaker"], "text": t["text"]} for t in conv.get(key, [])]
                sessions.append(ConversationSession(session_id=key, date=date, turns=turns))

            for qa in sample.get("qa", []):
                cat = qa.get("category")
                answer_text = qa.get("answer") or qa.get("adversarial_answer", "")
                cases.append(BenchmarkCase(
                    case_id=f"{sample_id}_q{len(cases)}",
                    question=qa["question"],
                    expected=ExpectedAnswer(text=answer_text),
                    evidence=EvidenceRef(),
                    sessions=sessions,
                    category=self.CATEGORY_NAMES.get(cat, str(cat)),
                    metadata={"sample_id": sample_id, "category_id": cat},
                ))

        return BenchmarkDataset(name="locomo", split=split, cases=cases)

    async def ingest(self, memory_core: Any, case: BenchmarkCase) -> int:
        count = 0
        for session in case.sessions:
            for turn in session.turns:
                content = f"[{session.date}] {turn['speaker']}: {turn['text']}" if session.date else f"{turn['speaker']}: {turn['text']}"
                await memory_core.remember(
                    content=content,
                    context={
                        "speaker": turn["speaker"],
                        "date": session.date,
                        "session_id": session.session_id,
                        "sample_id": case.metadata.get("sample_id"),
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

        by_cat: dict[str, list[float]] = {}
        for r in results:
            by_cat.setdefault(r.category or "unknown", []).append(r.score)

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "accuracy": _mean([r.score for r in results]),
            "total": len(results),
            "by_category": {cat: {"accuracy": _mean(scores), "count": len(scores)} for cat, scores in sorted(by_cat.items())},
        }
