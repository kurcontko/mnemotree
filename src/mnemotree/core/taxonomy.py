"""
Memory Type Taxonomy Utilities

Zero-cost convenience layer for episodic/semantic/procedural memory classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import MemoryType

if TYPE_CHECKING:
    from .memory import MemoryCore, RecallFilters


# ---------------------------------------------------------------------------
# Type Groups
# ---------------------------------------------------------------------------

EPISODIC_TYPES = [MemoryType.EPISODIC, MemoryType.AUTOBIOGRAPHICAL]
SEMANTIC_TYPES = [MemoryType.SEMANTIC]
PROCEDURAL_TYPES = [MemoryType.PROCEDURAL, MemoryType.PRIMING, MemoryType.CONDITIONING]


# ---------------------------------------------------------------------------
# Classification Functions
# ---------------------------------------------------------------------------


def is_episodic(memory_type: MemoryType) -> bool:
    """True for episodic/autobiographical (personal experience) memories."""
    return memory_type in EPISODIC_TYPES


def is_semantic(memory_type: MemoryType) -> bool:
    """True for semantic (factual knowledge) memories."""
    return memory_type in SEMANTIC_TYPES


def is_procedural(memory_type: MemoryType) -> bool:
    """True for procedural (skill/habit) memories."""
    return memory_type in PROCEDURAL_TYPES


# ---------------------------------------------------------------------------
# Query Helper Mixins (can be added to MemoryCore or used standalone)
# ---------------------------------------------------------------------------


class TaxonomyQueryMixin:
    """
    Mixin to add taxonomy-aware query methods to MemoryCore.

    Usage:
        # Option 1: Use filters directly
        await memory.recall(query, filters=RecallFilters(memory_types=EPISODIC_TYPES))

        # Option 2: Use this mixin's convenience methods
        await memory.recall_episodic(query)
    """

    async def recall_episodic(
        self,
        query: str,
        *,
        limit: int | None = None,
        **kwargs,
    ):
        """Recall episodic/autobiographical memories (personal experiences)."""
        from .memory import RecallFilters

        filters: RecallFilters = kwargs.pop("filters", RecallFilters())
        filters = RecallFilters(
            memory_types=EPISODIC_TYPES,
            tags=filters.tags,
            min_importance=filters.min_importance,
            max_importance=filters.max_importance,
            since=filters.since,
            until=filters.until,
            source=filters.source,
            author=filters.author,
            conversation_id=filters.conversation_id,
            user_id=filters.user_id,
        )
        return await self.recall(query, limit=limit, filters=filters, **kwargs)  # type: ignore

    async def recall_semantic(
        self,
        query: str,
        *,
        limit: int | None = None,
        **kwargs,
    ):
        """Recall semantic memories (facts, general knowledge)."""
        from .memory import RecallFilters

        filters: RecallFilters = kwargs.pop("filters", RecallFilters())
        filters = RecallFilters(
            memory_types=SEMANTIC_TYPES,
            tags=filters.tags,
            min_importance=filters.min_importance,
            max_importance=filters.max_importance,
            since=filters.since,
            until=filters.until,
            source=filters.source,
            author=filters.author,
            conversation_id=filters.conversation_id,
            user_id=filters.user_id,
        )
        return await self.recall(query, limit=limit, filters=filters, **kwargs)  # type: ignore

    async def recall_procedural(
        self,
        query: str,
        *,
        limit: int | None = None,
        **kwargs,
    ):
        """Recall procedural memories (skills, habits, learned behaviors)."""
        from .memory import RecallFilters

        filters: RecallFilters = kwargs.pop("filters", RecallFilters())
        filters = RecallFilters(
            memory_types=PROCEDURAL_TYPES,
            tags=filters.tags,
            min_importance=filters.min_importance,
            max_importance=filters.max_importance,
            since=filters.since,
            until=filters.until,
            source=filters.source,
            author=filters.author,
            conversation_id=filters.conversation_id,
            user_id=filters.user_id,
        )
        return await self.recall(query, limit=limit, filters=filters, **kwargs)  # type: ignore


# ---------------------------------------------------------------------------
# Standalone Query Helpers (for when you don't want to monkey-patch)
# ---------------------------------------------------------------------------


async def get_episodic_memories(
    memory_core: MemoryCore,
    query: str,
    *,
    limit: int | None = None,
    **kwargs,
):
    """Standalone helper to recall episodic memories."""
    from .memory import RecallFilters

    filters = kwargs.pop("filters", None) or RecallFilters()
    filters = RecallFilters(
        memory_types=EPISODIC_TYPES,
        tags=filters.tags,
        min_importance=filters.min_importance,
        max_importance=filters.max_importance,
        since=filters.since,
        until=filters.until,
        source=filters.source,
        author=filters.author,
        conversation_id=filters.conversation_id,
        user_id=filters.user_id,
    )
    return await memory_core.recall(query, limit=limit, filters=filters, **kwargs)


async def get_semantic_memories(
    memory_core: MemoryCore,
    query: str,
    *,
    limit: int | None = None,
    **kwargs,
):
    """Standalone helper to recall semantic memories."""
    from .memory import RecallFilters

    filters = kwargs.pop("filters", None) or RecallFilters()
    filters = RecallFilters(
        memory_types=SEMANTIC_TYPES,
        tags=filters.tags,
        min_importance=filters.min_importance,
        max_importance=filters.max_importance,
        since=filters.since,
        until=filters.until,
        source=filters.source,
        author=filters.author,
        conversation_id=filters.conversation_id,
        user_id=filters.user_id,
    )
    return await memory_core.recall(query, limit=limit, filters=filters, **kwargs)


async def get_procedural_memories(
    memory_core: MemoryCore,
    query: str,
    *,
    limit: int | None = None,
    **kwargs,
):
    """Standalone helper to recall procedural memories."""
    from .memory import RecallFilters

    filters = kwargs.pop("filters", None) or RecallFilters()
    filters = RecallFilters(
        memory_types=PROCEDURAL_TYPES,
        tags=filters.tags,
        min_importance=filters.min_importance,
        max_importance=filters.max_importance,
        since=filters.since,
        until=filters.until,
        source=filters.source,
        author=filters.author,
        conversation_id=filters.conversation_id,
        user_id=filters.user_id,
    )
    return await memory_core.recall(query, limit=limit, filters=filters, **kwargs)
