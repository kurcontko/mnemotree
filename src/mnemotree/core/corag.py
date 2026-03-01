"""Chain-of-Retrieval Augmented Generation (CoRAG).

Based on CoRAG (NeurIPS 2025): iteratively retrieve → LLM reason → refine query → retrieve.
Each hop the LLM inspects accumulated evidence and either proposes a new sub-query
or signals that retrieval is complete.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import MemoryItem
    from .retrieval import Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Protocol
# ---------------------------------------------------------------------------


@dataclass
class CoRAGConfig:
    """Configuration for ChainOfRetrieval."""

    max_hops: int = 4
    """Maximum number of iterative retrieval hops."""

    min_new_memories: int = 1
    """Stop early if a hop adds fewer than this many new memories."""

    merge_strategy: Literal["union", "weighted"] = "weighted"
    """How to merge memories across hops (currently both keep all, weighted reranks)."""

    candidate_multiplier: int = 2
    """Multiply `limit` by this when fetching candidates at each hop."""


@runtime_checkable
class ReasoningHook(Protocol):
    """Callable that inspects accumulated memories and returns a refined sub-query."""

    async def reason_and_refine(
        self,
        original_query: str,
        hop_index: int,
        accumulated: list[MemoryItem],
    ) -> str | None:
        """Return a refined sub-query string, or None to stop.

        Args:
            original_query: The user's original query string.
            hop_index: Zero-based index of the current hop.
            accumulated: All memories gathered so far.

        Returns:
            A new retrieval sub-query string, or None if retrieval is complete.
        """
        ...


# ---------------------------------------------------------------------------
# Default LLM reasoning hook (OpenAI-compatible)
# ---------------------------------------------------------------------------


@dataclass
class OpenAIReasoningHook:
    """Uses any OpenAI-compatible endpoint to produce the next sub-query.

    Compatible with Anthropic (via openai SDK + `base_url`), OpenAI, and
    locally-served models such as Ollama.
    """

    model: str = "claude-sonnet-4-6"
    base_url: str = "https://api.anthropic.com/v1"
    api_key: str = ""
    max_tokens: int = 256

    _SYSTEM = (
        "You are a memory retrieval agent. "
        "You will be shown an original query and memories retrieved so far. "
        "Identify ONE specific missing piece of information and write a concise retrieval query for it. "
        "If retrieval is complete, reply only: STOP"
    )

    _USER = (
        "Original query: {query}\n\n"
        "Retrieved so far ({n} memories):\n{summaries}\n\n"
        "What is the next retrieval query? (Reply STOP if done)"
    )

    async def reason_and_refine(
        self,
        original_query: str,
        hop_index: int,
        accumulated: list[MemoryItem],
    ) -> str | None:
        try:
            from openai import AsyncOpenAI
        except ImportError as err:
            raise ImportError(
                "OpenAIReasoningHook requires openai. Install with `pip install openai`."
            ) from err

        api_key = (
            self.api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )
        client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)

        summaries = "\n".join(
            f"- [{i+1}] {m.content[:200]}" for i, m in enumerate(accumulated[:15])
        )

        messages = [
            {"role": "system", "content": self._SYSTEM},
            {
                "role": "user",
                "content": self._USER.format(
                    query=original_query,
                    n=len(accumulated),
                    summaries=summaries or "(none)",
                ),
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("OpenAIReasoningHook LLM call failed at hop %d", hop_index)
            return None

        if "STOP" in text.upper():
            return None
        return text or None


# ---------------------------------------------------------------------------
# Chain-of-Retrieval engine
# ---------------------------------------------------------------------------


class ChainOfRetrieval:
    """Iterative retrieval loop driven by a ReasoningHook.

    Usage::

        hook = OpenAIReasoningHook(api_key="sk-...")
        cor = ChainOfRetrieval(retriever=mc.retrieval, hook=hook)
        memories, trace = await cor.retrieve("Who invented the transistor?", limit=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        hook: ReasoningHook,
        cfg: CoRAGConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.hook = hook
        self.cfg = cfg or CoRAGConfig()

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> tuple[list[MemoryItem], list[str]]:
        """Run chain-of-retrieval.

        Args:
            query: Original user query.
            limit: Maximum memories to return.

        Returns:
            Tuple of (memories, reasoning_trace) where trace is the list of
            sub-queries generated by the hook at each hop.
        """
        cfg = self.cfg
        accumulated: dict[str, MemoryItem] = {}
        trace: list[str] = []
        current_query = query
        candidate_limit = limit * cfg.candidate_multiplier

        for hop in range(cfg.max_hops):
            new_memories = await self.retriever.recall(
                current_query,
                limit=candidate_limit,
                scoring=True,
                update_access=False,
            )

            added = {
                m.memory_id: m for m in new_memories if m.memory_id not in accumulated
            }
            accumulated.update(added)

            logger.debug(
                "CoRAG hop=%d query=%r new=%d total=%d",
                hop,
                current_query,
                len(added),
                len(accumulated),
            )

            if len(added) < cfg.min_new_memories:
                logger.debug("CoRAG stopping: added %d < min_new_memories=%d", len(added), cfg.min_new_memories)
                break

            refined = await self.hook.reason_and_refine(
                original_query=query,
                hop_index=hop,
                accumulated=list(accumulated.values()),
            )
            if refined is None:
                logger.debug("CoRAG stopping: hook returned None at hop %d", hop)
                break

            trace.append(refined)
            current_query = refined

        results = _rank_and_limit(list(accumulated.values()), limit)
        return results, trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_and_limit(memories: list[MemoryItem], limit: int) -> list[MemoryItem]:
    """Sort by importance desc, then truncate to limit."""
    memories.sort(key=lambda m: m.importance, reverse=True)
    return memories[:limit]
