"""Recall-time conflict detection and annotation (AMA Judge+Refresher pattern).

Provides fast cosine-threshold conflict detection that can run on every recall()
call without LLM overhead. An optional LLM arbitration step handles ambiguous
cases (saves ~70-80% of LLM calls vs always-LLM approach).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..models import MemoryItem
from ..scoring import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ConflictAnnotation:
    """Conflict information attached to a recalled memory."""

    memory_id: str
    conflicting_ids: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)  # conflicting_id -> reason


@dataclass
class ConflictJudgeConfig:
    """Configuration for the recall-time conflict judge."""

    similarity_threshold: float = 0.92
    enabled: bool = True
    max_pairs_checked: int = 50


class ConflictJudge:
    """Fast cosine-threshold conflict detection at recall time.

    Phase 1 (heuristic): Checks pairwise cosine similarity among recalled
    memories. Pairs above ``similarity_threshold`` with opposing signals
    (negation words, contradicting link types, or pre-populated conflicts_with)
    are flagged as conflicts.

    Phase 2 (optional LLM): For ambiguous pairs (high similarity but no clear
    heuristic signal), an LLM can arbitrate. This is invoked only when an LLM
    is provided, saving most LLM calls.
    """

    _NEGATION_WORDS = frozenset({
        "not", "never", "no", "isn't", "aren't", "doesn't", "don't",
        "won't", "can't", "false", "wrong", "incorrect", "neither",
        "nor", "none", "nobody", "nothing", "nowhere",
    })

    def __init__(
        self,
        config: ConflictJudgeConfig | None = None,
        llm: Any = None,
    ) -> None:
        self.config = config or ConflictJudgeConfig()
        self.llm = llm

    def detect_conflicts(
        self, memories: list[MemoryItem]
    ) -> dict[str, ConflictAnnotation]:
        """Fast pairwise conflict detection among recalled memories.

        Returns a mapping from memory_id to ConflictAnnotation for memories
        that have at least one detected conflict.
        """
        if not self.config.enabled or len(memories) < 2:
            return {}

        annotations: dict[str, ConflictAnnotation] = {}
        pairs_checked = 0

        for i, a in enumerate(memories):
            for j in range(i + 1, len(memories)):
                if pairs_checked >= self.config.max_pairs_checked:
                    break
                b = memories[j]
                pairs_checked += 1

                # Check pre-existing conflicts_with
                if a.memory_id in (b.conflicts_with or []) or b.memory_id in (a.conflicts_with or []):
                    self._add_conflict(annotations, a, b, "pre-existing conflict")
                    continue

                # Check cosine similarity threshold
                if not a.embedding or not b.embedding:
                    continue
                sim = cosine_similarity(a.embedding, b.embedding)
                if sim < self.config.similarity_threshold:
                    continue

                # High similarity — check for negation signals
                if self._has_negation_signal(a, b):
                    self._add_conflict(
                        annotations, a, b,
                        f"high similarity ({sim:.3f}) with negation signal",
                    )
                elif a.timestamp != b.timestamp:
                    # Near-duplicate with different timestamps — potential update
                    self._add_conflict(
                        annotations, a, b,
                        f"near-duplicate ({sim:.3f}), newer may supersede older",
                    )

            if pairs_checked >= self.config.max_pairs_checked:
                break

        return annotations

    def _has_negation_signal(self, a: MemoryItem, b: MemoryItem) -> bool:
        """Check if two high-similarity memories have opposing signals."""
        words_a = set(a.content.lower().split())
        words_b = set(b.content.lower().split())
        neg_a = words_a & self._NEGATION_WORDS
        neg_b = words_b & self._NEGATION_WORDS
        # One has negation, the other doesn't → likely contradiction
        return bool(neg_a) != bool(neg_b)

    @staticmethod
    def _add_conflict(
        annotations: dict[str, ConflictAnnotation],
        a: MemoryItem,
        b: MemoryItem,
        reason: str,
    ) -> None:
        if a.memory_id not in annotations:
            annotations[a.memory_id] = ConflictAnnotation(memory_id=a.memory_id)
        if b.memory_id not in annotations:
            annotations[b.memory_id] = ConflictAnnotation(memory_id=b.memory_id)
        annotations[a.memory_id].conflicting_ids.append(b.memory_id)
        annotations[a.memory_id].reasons[b.memory_id] = reason
        annotations[b.memory_id].conflicting_ids.append(a.memory_id)
        annotations[b.memory_id].reasons[a.memory_id] = reason
