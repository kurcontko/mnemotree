"""ExtractorInference — local SmolLM2-based structured memory extractor."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .formatting import (
    EXTRACTOR_SYSTEM_PROMPT,
    build_extractor_record,
    format_extractor_user_message,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractorResult:
    """Output of a structured extraction."""

    raw_output: str
    parsed: dict[str, Any] | None
    memory_type: str
    confidence: float


class ExtractorInference:
    """Lazy-loading wrapper around a fine-tuned SmolLM2 extraction model.

    The model and tokenizer are loaded on first call to :meth:`extract`,
    keeping construction cheap.

    Parameters
    ----------
    model_path:
        Path to the saved model directory.
    device:
        Torch device string.
    max_new_tokens:
        Maximum tokens to generate during extraction.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        max_new_tokens: int = 512,
    ) -> None:
        self._model_path = str(model_path)
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading extractor model from %s onto %s", self._model_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
        )
        self._model.to(self._device)
        self._model.eval()
        logger.info(
            "Extractor model loaded (%s parameters)",
            sum(p.numel() for p in self._model.parameters()),
        )

    # ------------------------------------------------------------------
    # Sync core
    # ------------------------------------------------------------------

    def _extract_sync(
        self,
        content: str,
        memory_type: str,
        context: dict[str, Any] | None = None,
    ) -> ExtractorResult:
        import torch

        self._ensure_loaded()

        record = build_extractor_record(content, memory_type, context)
        user_message = format_extractor_user_message(record)

        messages = [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, input_len:]
        raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        parsed, confidence = self._parse_output(raw_output)

        return ExtractorResult(
            raw_output=raw_output,
            parsed=parsed,
            memory_type=memory_type,
            confidence=confidence,
        )

    @staticmethod
    def _parse_output(raw: str) -> tuple[dict[str, Any] | None, float]:
        """Try to parse the model output as JSON and derive a confidence score."""
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return None, 0.0
            confidence = parsed.pop("confidence", None)
            if isinstance(confidence, (int, float)):
                return parsed, float(min(max(confidence, 0.0), 1.0))
            return parsed, 0.8  # valid JSON but no explicit confidence
        except (json.JSONDecodeError, TypeError):
            return None, 0.0

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def extract(
        self,
        content: str,
        memory_type: str,
        context: dict[str, Any] | None = None,
    ) -> ExtractorResult:
        """Extract structured data from *content* given its *memory_type*.

        Runs torch inference in a thread pool so it doesn't block the event loop.
        """
        return await asyncio.to_thread(self._extract_sync, content, memory_type, context)
