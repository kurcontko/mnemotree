from __future__ import annotations

from typing import Any

from .models import SummaryResult

_SUMMARY_SYSTEM = """Generate a concise summary of the chat interaction.
Capture: key topics discussed, main decisions made, action items, and entities involved.
Present in a clear, structured format suitable for semantic search."""

_STRUCTURED_SYSTEM = """Generate a concise summary of the chat interaction.
Capture: key topics discussed, main decisions made, action items, and entities involved.
Return a JSON object matching the SummaryResult schema."""


class Summarizer:
    """Generates summaries of memory content."""

    def __init__(self, model: str | Any = "openai:gpt-4o-mini"):
        self._model = model
        self._text_agent: Any = None
        self._structured_agent: Any = None

    def _ensure_agents(self) -> None:
        if self._text_agent is None:
            from pydantic_ai import Agent

            self._text_agent = Agent(self._model, output_type=str, instructions=_SUMMARY_SYSTEM)
            self._structured_agent = Agent(
                self._model, output_type=SummaryResult, instructions=_STRUCTURED_SYSTEM
            )

    async def summarize(self, content: str, context: str = "", format: str = "text") -> str | dict:
        """Generate a summary of the chat interaction."""
        self._ensure_agents()
        user_prompt = f"Content: {content}\n\nContext: {context}"
        if format == "structured":
            result = await self._structured_agent.run(user_prompt)
            return result.output.model_dump()
        result = await self._text_agent.run(user_prompt)
        return result.output
