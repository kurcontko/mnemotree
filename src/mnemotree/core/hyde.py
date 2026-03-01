"""HyDE (Hypothetical Document Embeddings) for improved recall.

Instead of embedding the raw query, generate a hypothetical ideal memory via LLM
and embed that. This bridges the lexical gap between queries and stored memories.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HyDEEmbedder"]

_HYDE_PROMPT = """\
Given the following query about a memory, write a single short passage that would be
the ideal memory matching this query. Write it as if you are recalling the memory.
Query: {query}
Ideal memory:""".strip()


class HyDEEmbedder:
    """Wraps an embedder to use hypothetical document embeddings instead of raw query."""

    def __init__(self, llm: Any, embedder: Any) -> None:
        self.llm = llm
        self.embedder = embedder

    async def aembed_query(self, query: str) -> list[float]:
        hypothetical = await self._generate(query)
        return await self.embedder.aembed_query(hypothetical or query)

    async def _generate(self, query: str) -> str:
        response = await self.llm.ainvoke(_HYDE_PROMPT.format(query=query))
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()
