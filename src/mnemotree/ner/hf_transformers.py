from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseNER, NERResult


class TransformersNER(BaseNER):
    """Hugging Face Transformers token-classification NER (e.g. DistilBERT NER)."""

    def __init__(
        self,
        *,
        model: str = "dslim/distilbert-NER",
        tokenizer: str | None = None,
        aggregation_strategy: str = "simple",
        device: int = -1,
        pipeline_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            model: Model id/path compatible with `transformers.pipeline`.
            tokenizer: Optional tokenizer id/path (defaults to `model`).
            aggregation_strategy: Group sub-tokens into full entities (recommended).
            device: -1 for CPU, or GPU index (e.g. 0) if available.
            pipeline_kwargs: Extra kwargs passed to `transformers.pipeline`.
        """
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "TransformersNER requires optional dependencies. "
                "Install with: pip install 'mnemotree[ner_hf]'"
            ) from exc

        kwargs: dict[str, Any] = dict(pipeline_kwargs or {})
        try:
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer or model,
                aggregation_strategy=aggregation_strategy,
                device=device,
                **kwargs,
            )
        except TypeError:
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer or model,
                grouped_entities=True,
                device=device,
                **kwargs,
            )

    async def extract_entities(self, text: str) -> NERResult:
        predictions: list[dict[str, Any]] = await asyncio.to_thread(self._pipeline, text)

        entities: dict[str, str] = {}
        mentions: dict[str, list[str]] = {}
        confidence: dict[str, float] = {}

        for pred in predictions:
            entity_text = (pred.get("word") or pred.get("text") or "").strip()
            if not entity_text:
                continue

            entity_type = (
                pred.get("entity_group") or pred.get("entity") or pred.get("label") or "ENTITY"
            )
            score = pred.get("score")
            try:
                score_f = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_f = None

            start = pred.get("start")
            end = pred.get("end")

            if (
                entity_text not in entities
                or score_f is not None
                and score_f > confidence.get(entity_text, -1.0)
            ):
                entities[entity_text] = str(entity_type)

            if score_f is not None:
                confidence[entity_text] = max(confidence.get(entity_text, 0.0), score_f)

            if isinstance(start, int) and isinstance(end, int):
                context = self._get_context(text, start, end)
                mentions.setdefault(entity_text, []).append(context)

        return NERResult(
            entities=entities,
            mentions=mentions,
            confidence=confidence or None,
        )
