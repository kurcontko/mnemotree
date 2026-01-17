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

    @staticmethod
    def _coerce_score(score: Any) -> float | None:
        try:
            return float(score) if score is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_span(start: Any, end: Any) -> tuple[int | None, int | None]:
        if isinstance(start, int) and isinstance(end, int):
            return start, end
        return None, None

    def _parse_prediction(
        self, text: str, pred: dict[str, Any]
    ) -> tuple[str, str, float | None, str | None] | None:
        entity_text = (pred.get("word") or pred.get("text") or "").strip()
        if not entity_text:
            return None

        entity_type = pred.get("entity_group") or pred.get("entity") or pred.get("label") or "ENTITY"
        score_f = self._coerce_score(pred.get("score"))
        start, end = self._coerce_span(pred.get("start"), pred.get("end"))
        context = self._get_context(text, start, end) if start is not None and end is not None else None
        return entity_text, str(entity_type), score_f, context

    @staticmethod
    def _should_update_entity(
        entity_text: str,
        score_f: float | None,
        confidence: dict[str, float],
        entities: dict[str, str],
    ) -> bool:
        if entity_text not in entities:
            return True
        return score_f is not None and score_f > confidence.get(entity_text, -1.0)

    @staticmethod
    def _update_confidence(
        entity_text: str,
        score_f: float | None,
        confidence: dict[str, float],
    ) -> None:
        if score_f is None:
            return
        confidence[entity_text] = max(confidence.get(entity_text, 0.0), score_f)

    async def extract_entities(self, text: str) -> NERResult:
        predictions: list[dict[str, Any]] = await asyncio.to_thread(self._pipeline, text)

        entities: dict[str, str] = {}
        mentions: dict[str, list[str]] = {}
        confidence: dict[str, float] = {}

        for pred in predictions:
            parsed = self._parse_prediction(text, pred)
            if parsed is None:
                continue
            entity_text, entity_type, score_f, context = parsed

            if self._should_update_entity(entity_text, score_f, confidence, entities):
                entities[entity_text] = entity_type

            self._update_confidence(entity_text, score_f, confidence)

            if context is not None:
                mentions.setdefault(entity_text, []).append(context)

        return NERResult(
            entities=entities,
            mentions=mentions,
            confidence=confidence or None,
        )
