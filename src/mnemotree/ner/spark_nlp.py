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
            data = item.asDict(recursive=True) if hasattr(item, "asDict") else item
            if not isinstance(data, dict):
                continue

            entity_text = str(data.get("result") or "").strip()
            if not entity_text:
                continue

            metadata = data.get("metadata") or {}
            entity_type = (
                metadata.get("entity") or metadata.get("label") or metadata.get("ner") or "ENTITY"
            )

            try:
                start = int(data.get("begin")) if data.get("begin") is not None else None
                end = int(data.get("end")) + 1 if data.get("end") is not None else None
            except (TypeError, ValueError):
                start, end = None, None

            score = metadata.get("confidence") or metadata.get("score")
            try:
                score_f = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_f = None

            entities[entity_text] = str(entity_type)
            if score_f is not None:
                confidence[entity_text] = max(confidence.get(entity_text, 0.0), score_f)

            if start is not None and end is not None:
                context = self._get_context(text, start, end)
                mentions.setdefault(entity_text, []).append(context)

        return NERResult(
            entities=entities,
            mentions=mentions,
            confidence=confidence or None,
        )
