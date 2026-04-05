"""Conversation ingestion adapter for mnemotree.

Provides a structured way to ingest multi-session conversations into
MemoryCore, with automatic coreference context, temporal anchoring,
and entity tracking.

Usage::

    from mnemotree.adapters.conversation import ConversationIngestor

    ingestor = ConversationIngestor(memory_core)
    stats = await ingestor.ingest(
        sessions=[
            Session(
                date="2023-05-07 13:56",
                turns=[
                    Turn(speaker="Alice", text="I went hiking yesterday."),
                    Turn(speaker="Bob", text="That sounds fun!"),
                ],
            ),
        ],
        metadata={"conversation_id": "conv-1"},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """A single conversation turn."""

    speaker: str
    text: str


@dataclass
class Session:
    """A conversation session with a date and sequence of turns."""

    date: str
    turns: list[Turn]
    session_id: str | None = None


@dataclass
class IngestionStats:
    """Statistics from a conversation ingestion run."""

    total_memories: int = 0
    total_sessions: int = 0
    speakers: list[str] = field(default_factory=list)


class ConversationIngestor:
    """Ingests multi-session conversations into a MemoryCore.

    Handles:
    - Speaker discovery for coreference resolution
    - Per-turn normalization context (speaker, other_speaker, reference_date)
    - Recent entity tracking across turns
    - Content formatting with timestamps
    """

    def __init__(
        self,
        memory: Any,
        *,
        content_template: str = "[{date}] {speaker}: {text}",
        default_memory_type: str | None = None,
    ) -> None:
        """
        Args:
            memory: A MemoryCore instance (with normalization configured).
            content_template: Format string for memory content.
                Available keys: ``date``, ``speaker``, ``text``.
            default_memory_type: If set, passed as ``memory_type`` to
                ``remember()``.  When ``None`` (the default), the router /
                analyzer chooses the type automatically.
        """
        self._memory = memory
        self._content_template = content_template
        self._default_memory_type = default_memory_type

    async def ingest(
        self,
        sessions: list[Session],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionStats:
        """Ingest a conversation (one or more sessions) into memory.

        Args:
            sessions: Chronologically ordered sessions to ingest.
            metadata: Extra metadata attached to every memory
                (e.g. ``conversation_id``, ``sample_id``).

        Returns:
            IngestionStats with counts and discovered speakers.
        """
        metadata = metadata or {}

        # Discover all speakers for coreference context.
        all_speakers: set[str] = set()
        for session in sessions:
            for turn in session.turns:
                all_speakers.add(turn.speaker)
        speaker_list = sorted(all_speakers)

        memory_count = 0
        recent_entities: list[str] = []

        for idx, session in enumerate(sessions):
            session_id = session.session_id or f"session_{idx}"

            for turn in session.turns:
                content = self._content_template.format(
                    date=session.date,
                    speaker=turn.speaker,
                    text=turn.text,
                )

                # Build normalization context for coref + temporal.
                other_speakers = [s for s in speaker_list if s != turn.speaker]
                other_speaker = (
                    other_speakers[0] if len(other_speakers) == 1 else None
                )

                context: dict[str, Any] = {
                    "speaker": turn.speaker,
                    "other_speaker": other_speaker,
                    "reference_date": session.date,
                    "recent_entities": list(recent_entities[-5:]),
                }

                remember_kwargs: dict[str, Any] = {
                    "content": content,
                    "context": context,
                    "metadata": {
                        **metadata,
                        "session": session_id,
                        "speaker": turn.speaker,
                        "date": session.date,
                    },
                }
                if self._default_memory_type is not None:
                    remember_kwargs["memory_type"] = self._default_memory_type

                await self._memory.remember(**remember_kwargs)
                memory_count += 1

                # Track recent entities (speaker names).
                if turn.speaker not in recent_entities:
                    recent_entities.append(turn.speaker)

        return IngestionStats(
            total_memories=memory_count,
            total_sessions=len(sessions),
            speakers=speaker_list,
        )
