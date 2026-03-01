from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent


T = TypeVar("T", bound=BaseModel)


class AnalysisStrategy(Enum):
    """Defines different strategies for memory analysis"""

    QUICK = "quick"
    DEEP = "deep"
    ADAPTIVE = "adaptive"


class BaseAnalyzer(ABC):
    """Base class for all analyzers. Uses PydanticAI for structured LLM output."""

    def __init__(self, model: str | Any = "openai:gpt-4o-mini"):
        self.model = model
        self._agent: Agent | None = None

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this analyzer."""

    @abstractmethod
    def _get_output_type(self) -> type[BaseModel]:
        """Return the Pydantic output model."""

    def _build_agent(self) -> Agent:
        return Agent(
            self.model,
            output_type=self._get_output_type(),
            instructions=self._get_system_prompt(),
        )

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    async def analyze(self, content: str, context_str: str) -> dict:
        """Perform analysis, returning a dict (model_dump of the output)."""
        user_prompt = f"Content: {content}\n\nContext: {context_str}"
        result = await self.agent.run(user_prompt)
        output = result.output
        return output.model_dump() if hasattr(output, "model_dump") else dict(output)
