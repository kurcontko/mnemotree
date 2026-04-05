"""Mastra OM-inspired Observer/Reflector pattern for background memory creation.

The Observer watches conversation turns and extracts memorable facts into an
append-only log — no per-turn retrieval needed (key Mastra OM insight that
achieves 94.87% LongMemEval SOTA).

The Reflector wraps the existing MemoryConsolidator + ConflictJudge to run
periodic consolidation cycles ("sleep processing").

Reference: Mastra Observational Memory — SOTA 94.87% LongMemEval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import MemoryItem, MemoryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
Extract memorable facts from this conversation turn. Return one fact per line.
Only include facts worth remembering long-term (preferences, decisions, events,
plans, relationships, knowledge). Skip greetings, filler, and transient details.
If nothing is worth remembering, return NONE.

Turn:
{content}"""


@dataclass
class ObservationConfig:
    """Configuration for the Observer."""

    min_content_length: int = 15
    max_observations_per_turn: int = 5
    default_memory_type: MemoryType = MemoryType.EPISODIC
    default_importance: float = 0.5
    use_llm_extraction: bool = True
    extract_temporal: bool = True


@dataclass
class Observation:
    """A single observation extracted from a conversation turn."""

    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    observation_date: datetime | None = None
    event_time: datetime | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "observer"


class MemoryObserver:
    """Extracts memorable facts from conversation turns.

    The Observer follows the Mastra OM pattern: write-only, no per-turn
    retrieval. Facts are appended to the memory store and periodically
    consolidated by the Reflector.

    Can operate in two modes:
    - **LLM mode** (``use_llm_extraction=True``): Uses an LLM to extract
      salient facts from each turn.
    - **Passthrough mode** (``use_llm_extraction=False``): Stores the raw
      content directly (useful for lite/edge deployments).
    """

    def __init__(
        self,
        llm: Any | None = None,
        config: ObservationConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or ObservationConfig()

    async def observe(
        self,
        content: str,
        *,
        context: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> list[Observation]:
        """Extract observations from a conversation turn.

        Args:
            content: The conversation turn text.
            context: Optional context dict (conversation_id, role, etc.).
            user_id: Optional user identifier for the observation.

        Returns:
            List of extracted observations. May be empty if nothing is
            worth remembering.
        """
        if len(content.strip()) < self.config.min_content_length:
            return []

        now = datetime.now(timezone.utc)

        if self.config.use_llm_extraction and self.llm is not None:
            observations = await self._llm_extract(content, now)
        else:
            observations = self._passthrough_extract(content, now)

        # Cap observations per turn
        return observations[: self.config.max_observations_per_turn]

    async def _llm_extract(self, content: str, now: datetime) -> list[Observation]:
        """Use LLM to extract memorable facts."""
        try:
            response = await self.llm.ainvoke(  # type: ignore[union-attr]
                _EXTRACT_PROMPT.format(content=content)
            )
            text = (response.content if hasattr(response, "content") else str(response)) or ""

            if not text.strip() or text.strip().upper() == "NONE":
                return []

            observations: list[Observation] = []
            for line in text.strip().splitlines():
                line = line.strip()
                # Strip leading numbering/bullets
                line = re.sub(r"^[\d\.\-\*\)\]]+\s*", "", line).strip()
                if not line or len(line) < self.config.min_content_length:
                    continue
                observations.append(
                    Observation(
                        content=line,
                        memory_type=self.config.default_memory_type,
                        importance=self.config.default_importance,
                        observation_date=now,
                    )
                )
            return observations
        except Exception:
            logger.debug("LLM observation extraction failed", exc_info=True)
            return self._passthrough_extract(content, now)

    def _passthrough_extract(self, content: str, now: datetime) -> list[Observation]:
        """Store content directly without LLM extraction."""
        return [
            Observation(
                content=content,
                memory_type=self.config.default_memory_type,
                importance=self.config.default_importance,
                observation_date=now,
            )
        ]


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------


@dataclass
class ReflectorConfig:
    """Configuration for the Reflector consolidation cycle."""

    min_memories_for_consolidation: int = 10
    consolidation_interval_hours: float = 24.0
    conflict_similarity_threshold: float = 0.92
    promote_episodic_to_semantic: bool = True
    deprecate_low_importance: bool = True
    deprecate_importance_threshold: float = 0.2


class MemoryReflector:
    """Periodic consolidation + conflict resolution.

    Wraps the existing MemoryConsolidator and ConflictJudge to run
    "sleep processing" cycles that:
    1. Cluster similar episodic observations
    2. Promote clusters to semantic memories
    3. Detect and annotate conflicts
    4. Deprecate low-signal observations

    This is the Reflector half of the Mastra OM pattern.
    """

    def __init__(
        self,
        config: ReflectorConfig | None = None,
    ) -> None:
        self.config = config or ReflectorConfig()
        self._last_run: datetime | None = None

    async def should_run(self, memory_count: int) -> bool:
        """Check if consolidation should run based on config thresholds."""
        if memory_count < self.config.min_memories_for_consolidation:
            return False
        if self._last_run is None:
            return True
        from datetime import timedelta

        elapsed = datetime.now(timezone.utc) - self._last_run
        return elapsed >= timedelta(hours=self.config.consolidation_interval_hours)

    async def run(
        self,
        memories: list[MemoryItem],
        *,
        consolidator: Any | None = None,
        conflict_judge: Any | None = None,
    ) -> ReflectorResult:
        """Run a consolidation cycle on the provided memories.

        Args:
            memories: Memories to consolidate (typically recent episodic).
            consolidator: Optional MemoryConsolidator instance.
            conflict_judge: Optional ConflictJudge instance.

        Returns:
            ReflectorResult with consolidation and conflict info.
        """
        self._last_run = datetime.now(timezone.utc)

        conflicts: dict[str, Any] = {}
        consolidation_result: Any = None

        # Phase 1: Conflict detection
        if conflict_judge is not None and memories:
            try:
                conflicts = conflict_judge.detect_conflicts(memories)
            except Exception:
                logger.debug("Conflict detection failed", exc_info=True)

        # Phase 2: Consolidation
        if consolidator is not None and memories:
            try:
                consolidation_result = await consolidator.consolidate(memories)
            except Exception:
                logger.debug("Consolidation failed", exc_info=True)

        return ReflectorResult(
            memories_processed=len(memories),
            conflicts_found=len(conflicts),
            conflict_annotations=conflicts,
            consolidation_result=consolidation_result,
            run_at=self._last_run,
        )


@dataclass
class ReflectorResult:
    """Result of a Reflector consolidation cycle."""

    memories_processed: int = 0
    conflicts_found: int = 0
    conflict_annotations: dict[str, Any] = field(default_factory=dict)
    consolidation_result: Any = None
    run_at: datetime | None = None
