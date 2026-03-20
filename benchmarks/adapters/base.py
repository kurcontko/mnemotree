"""Base protocol and data types for benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ConversationSession:
    """A single conversation session with ordered turns."""

    session_id: str
    date: str | None = None
    turns: list[dict[str, str]] = field(default_factory=list)


@dataclass
class EvidenceRef:
    """Reference to ground-truth evidence for a question."""

    session_ids: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    text_snippets: list[str] = field(default_factory=list)


@dataclass
class ExpectedAnswer:
    """Expected answer with optional alternatives and type info."""

    text: str
    alternatives: list[str] = field(default_factory=list)
    answer_type: str = "free_text"  # free_text, multiple_choice, yes_no, abstain


@dataclass
class BenchmarkCase:
    """A single evaluation case (question + evidence + expected answer)."""

    case_id: str
    question: str
    expected: ExpectedAnswer
    evidence: EvidenceRef = field(default_factory=EvidenceRef)
    sessions: list[ConversationSession] = field(default_factory=list)
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """Result of evaluating a single benchmark case."""

    case_id: str
    question: str
    predicted: str
    expected: str
    score: float  # 0.0 to 1.0
    category: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """Loaded benchmark dataset ready for evaluation."""

    name: str
    split: str
    cases: list[BenchmarkCase]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Protocol that all benchmark adapters must implement."""

    name: str

    def load(self, split: str, data_dir: str | None = None) -> BenchmarkDataset:
        """Load and parse benchmark data for a given split."""
        ...

    async def ingest(self, memory_core: Any, case: BenchmarkCase) -> int:
        """Ingest evidence/sessions into mnemotree. Returns number of memories stored."""
        ...

    async def run_case(self, memory_core: Any, case: BenchmarkCase, **kwargs: Any) -> CaseResult:
        """Run a single evaluation case: retrieve + generate + score."""
        ...

    def aggregate(self, results: list[CaseResult]) -> dict[str, Any]:
        """Aggregate per-case results into summary metrics."""
        ...
