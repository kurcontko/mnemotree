"""RouterInference — local ModernBERT-based memory-type classifier."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .formatting import ROUTER_LABEL_NAMES, build_router_record, format_router_input

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.55


@dataclass
class RouterResult:
    """Output of a router classification."""

    primary_type: str
    probabilities: dict[str, float]
    active_labels: list[str]


class RouterInference:
    """Lazy-loading wrapper around a fine-tuned ModernBERT multi-label classifier.

    The model and tokenizer are loaded on first call to :meth:`classify`,
    keeping construction cheap.

    Parameters
    ----------
    model_path:
        Path to the saved model directory (contains ``config.json``, weights, etc.).
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...).
    threshold:
        Minimum sigmoid probability to consider a label *active*.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._model_path = str(model_path)
        self._device = device
        self._threshold = threshold
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading router model from %s onto %s", self._model_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path)
        self._model.to(self._device)
        self._model.eval()
        logger.info("Router model loaded (%s parameters)", sum(p.numel() for p in self._model.parameters()))

    # ------------------------------------------------------------------
    # Sync core
    # ------------------------------------------------------------------

    def _classify_sync(self, text: str) -> RouterResult:
        import torch

        self._ensure_loaded()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, num_labels)

        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()  # list[float]

        probabilities = {
            label: round(prob, 4) for label, prob in zip(ROUTER_LABEL_NAMES, probs)
        }
        active_labels = [
            label for label, prob in probabilities.items() if prob >= self._threshold
        ]

        # Pick highest-probability label as the primary type.
        primary_type = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]

        return RouterResult(
            primary_type=primary_type,
            probabilities=probabilities,
            active_labels=active_labels,
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def classify(self, text: str) -> RouterResult:
        """Classify *text* into a memory type.

        Runs the torch inference in a thread pool so it doesn't block the
        event loop.
        """
        return await asyncio.to_thread(self._classify_sync, text)

    async def classify_record(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> RouterResult:
        """Convenience wrapper that formats ``remember()`` args before classifying."""
        record = build_router_record(content, context)
        text = format_router_input(record)
        return await self.classify(text)
