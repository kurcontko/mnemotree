"""LongMemEval benchmark adapter.

Covers: information extraction, multi-session reasoning,
knowledge updates, temporal reasoning, abstention.

Splits: oracle, s_cleaned, m_cleaned
Source: https://github.com/xiaowu0162/LongMemEval

Expected data format (longmemeval_{split}.json):
[
  {
    "question_id": "...",
    "question": "...",
    "answer": "...",
    "haystack_sessions": [
      {"session_id": "...", "date": "...", "turns": [{"role": "user"|"assistant", "content": "..."}]}
    ],
    "answer_session_ids": ["session_1", "session_5"],
    "question_type": "information_extraction"|"multi_session"|"knowledge_update"|"temporal"|"abstention"
  }
]
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
    EvidenceRef,
    ExpectedAnswer,
)
from benchmarks.lib.metrics import mean, session_recall

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "mnemotree-multi-bench" / "benchmarks" / "datasets"

SPLIT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "s_cleaned": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "m_cleaned": "longmemeval_m_cleaned.json",
}


class LongMemEvalAdapter:
    """Adapter for LongMemEval (Long-term Memory Evaluation)."""

    name = "longmemeval"

    def load(self, split: str = "oracle", data_dir: str | None = None) -> BenchmarkDataset:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        filename = SPLIT_FILES.get(split, f"longmemeval_{split}.json")
        data_path = base / filename

        if not data_path.exists():
            raise FileNotFoundError(
                f"LongMemEval dataset not found at {data_path}. "
                f"Download from https://github.com/xiaowu0162/LongMemEval and place in {base}/"
            )

        with data_path.open() as f:
            data = json.load(f)

        cases: list[BenchmarkCase] = []
        for item in data:
            qid = item.get("question_id", f"q{len(cases)}")
            qtype = item.get("question_type", "unknown")

            sessions = []
            for sess in item.get("haystack_sessions", []):
                turns = []
                for turn in sess.get("turns", []):
                    turns.append({
                        "speaker": turn.get("role", "user"),
                        "text": turn.get("content", ""),
                    })
                sessions.append(ConversationSession(
                    session_id=sess.get("session_id", f"s{len(sessions)}"),
                    date=sess.get("date"),
                    turns=turns,
                ))

            answer_session_ids = item.get("answer_session_ids", [])
            answer_text = item.get("answer", "")

            # Determine answer type based on question_type
            answer_type = "free_text"
            if qtype == "abstention":
                answer_type = "abstain"

            cases.append(BenchmarkCase(
                case_id=qid,
                question=item["question"],
                expected=ExpectedAnswer(text=answer_text, answer_type=answer_type),
                evidence=EvidenceRef(session_ids=answer_session_ids),
                sessions=sessions,
                category=qtype,
                metadata={
                    "question_id": qid,
                    "answer_session_ids": answer_session_ids,
                },
            ))

        return BenchmarkDataset(name="longmemeval", split=split, cases=cases)

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

        # Compute session recall: which expected sessions appear in retrieved memories
        expected_sids = set(case.metadata.get("answer_session_ids", []))
        retrieved_sids = set()
        for m in retrieved:
            ctx = getattr(m, "context", None) or {}
            sid = ctx.get("session_id") if isinstance(ctx, dict) else None
            if sid:
                retrieved_sids.add(sid)
            md = getattr(m, "metadata", None) or {}
            sid2 = md.get("session_id") if isinstance(md, dict) else None
            if sid2:
                retrieved_sids.add(sid2)

        sess_recall = session_recall(retrieved_sids, expected_sids)

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
            metadata={"session_recall": sess_recall},
        )

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        if not results:
            return {"accuracy": 0.0, "total": 0}

        by_cat: dict[str, list[CaseResult]] = {}
        for r in results:
            by_cat.setdefault(r.category or "unknown", []).append(r)

        sess_recalls = [r.metadata.get("session_recall", 0.0) for r in results]

        return {
            "accuracy": mean([r.score for r in results]),
            "session_recall": mean(sess_recalls),
            "total": len(results),
            "by_category": {
                cat: {
                    "accuracy": mean([r.score for r in cat_results]),
                    "session_recall": mean([r.metadata.get("session_recall", 0.0) for r in cat_results]),
                    "count": len(cat_results),
                }
                for cat, cat_results in sorted(by_cat.items())
            },
        }
