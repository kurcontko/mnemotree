from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseNER, NERResult


class SparkNLPNER(BaseNER):
    """
    Spark NLP NER backend.

    This wrapper is intentionally minimal: you provide a configured Spark NLP pipeline model
    that outputs an entity "chunk" column (commonly produced by NerConverter).
    """

    def __init__(
        self,
        *,
        spark: Any,
        pipeline_model: Any,
        input_col: str = "text",
        output_col: str = "ner_chunk",
    ):
        try:
            import pyspark  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SparkNLPNER requires optional dependencies (pyspark + spark-nlp). "
                "Install with: pip install 'mnemotree[ner_spark]'"
            ) from exc

        self._spark = spark
        self._pipeline_model = pipeline_model
        self._input_col = input_col
        self._output_col = output_col

    @staticmethod
    def _item_to_dict(item: Any) -> dict[str, Any] | None:
        data = item.asDict(recursive=True) if hasattr(item, "asDict") else item
        return data if isinstance(data, dict) else None

    @staticmethod
    def _coerce_span(begin: Any, end: Any) -> tuple[int | None, int | None]:
        try:
            start = int(begin) if begin is not None else None
            end_val = int(end) + 1 if end is not None else None
        except (TypeError, ValueError):
            return None, None
        return start, end_val

    @staticmethod
    def _coerce_score(score: Any) -> float | None:
        try:
            return float(score) if score is not None else None
        except (TypeError, ValueError):
            return None

    def _parse_item(self, text: str, item: Any) -> tuple[str, str, float | None, str | None] | None:
        data = self._item_to_dict(item)
        if data is None:
            return None

        entity_text = str(data.get("result") or "").strip()
        if not entity_text:
            return None

        metadata = data.get("metadata") or {}
        entity_type = (
            metadata.get("entity") or metadata.get("label") or metadata.get("ner") or "ENTITY"
        )
        start, end = self._coerce_span(data.get("begin"), data.get("end"))
        score_f = self._coerce_score(metadata.get("confidence") or metadata.get("score"))
        context = (
            self._get_context(text, start, end) if start is not None and end is not None else None
        )
        return entity_text, str(entity_type), score_f, context

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
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> NERResult:
        df = self._spark.createDataFrame([(text,)], schema=[self._input_col])
        out = self._pipeline_model.transform(df).select(self._output_col).first()
        raw = out[0] if out is not None else []

        entities: dict[str, str] = {}
        mentions: dict[str, list[str]] = {}
        confidence: dict[str, float] = {}

        for item in raw or []:
            parsed = self._parse_item(text, item)
            if parsed is None:
                continue
            entity_text, entity_type, score_f, context = parsed

            entities[entity_text] = entity_type
            self._update_confidence(entity_text, score_f, confidence)

            if context is not None:
                mentions.setdefault(entity_text, []).append(context)

        return NERResult(
            entities=entities,
            mentions=mentions,
            confidence=confidence or None,
        )
