"""Answer generation and LLM-as-judge scoring.

Extracted from mnemotree_locomo_optimized.py for reuse across adapters.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


def get_openai_client() -> tuple[OpenAI, str]:
    """Return (client, model) from environment."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-5.2")
    return client, model


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
async def generate_answer(
    client: OpenAI,
    model: str,
    context: str,
    question: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str:
    """Generate an answer given retrieved context and a question."""
    if system_prompt is None:
        system_prompt = (
            "You are answering questions about a conversation history. "
            "Use only the provided memories to answer. Be concise and specific."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Memories:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
async def judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    expected: str,
    predicted: str,
) -> float:
    """Score a predicted answer against expected using LLM judge.

    Returns: 1.0 (correct), 0.5 (partial), 0.0 (incorrect).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise answer evaluator. Compare the predicted answer "
                    "to the reference answer and respond with exactly one word: "
                    "correct, partial, or incorrect."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Reference answer: {expected}\n"
                    f"Predicted answer: {predicted}"
                ),
            },
        ],
        temperature=0,
        max_tokens=100,  # Reasoning models need more tokens
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    if "correct" in verdict and "incorrect" not in verdict:
        return 1.0
    if "partial" in verdict:
        return 0.5
    return 0.0
