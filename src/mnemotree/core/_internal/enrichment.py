from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...analysis.keywords import KeywordExtractor
from ...analysis.memory_analyzer import MemoryAnalyzer
from ...analysis.models import MemoryAnalysisResult
from ...analysis.summarizer import Summarizer
from ...ner.base import BaseNER
from ...ner.spacy import NERResult

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    embedding: list[float]
    summary: str | None = None
    analysis: MemoryAnalysisResult | None = None
    keywords: list[str] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    entity_mentions: dict[str, list[str]] = field(default_factory=dict)


@runtime_checkable
class EnrichmentPipeline(Protocol):
    async def enrich(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        analyze: bool = False,
        summarize: bool = False,
        *,
        flags: dict[str, Any] | None = None,
    ) -> EnrichmentResult: ...


class StandardEnrichmentPipeline:
    def __init__(
        self,
        embedder: Any,
        ner: BaseNER | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        analyzer: MemoryAnalyzer | None = None,
        summarizer: Summarizer | None = None,
    ):
        self.embedder = embedder
        self.ner = ner
        self.keyword_extractor = keyword_extractor
        self.analyzer = analyzer
        self.summarizer = summarizer

    async def enrich(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        analyze: bool = False,
        summarize: bool = False,
        *,
        flags: dict[str, Any] | None = None,
    ) -> EnrichmentResult:
        started = time.perf_counter()
        if flags:
            analyze = flags.get("analyze", analyze)
            summarize = flags.get("summarize", summarize)
        if summarize and not self.summarizer:
            raise RuntimeError("Summarizer not configured.")
        if analyze and not self.analyzer:
            raise RuntimeError("Analyzer not configured.")

        logger.debug(
            "enrich start summarize=%s analyze=%s ner=%s keywords=%s",
            summarize,
            analyze,
            bool(self.ner),
            bool(self.keyword_extractor),
        )

        # Prepare tasks
        tasks: dict[str, asyncio.Task] = {
            "embedding": asyncio.create_task(self._get_embedding(content)),
        }
        if self.ner:
            tasks["ner"] = asyncio.create_task(self.ner.extract_entities(content))
        if self.keyword_extractor:
            tasks["keywords"] = asyncio.create_task(self.keyword_extractor.extract(content))
        if summarize:
            assert self.summarizer is not None
            tasks["summary"] = asyncio.create_task(self.summarizer.summarize(content))
        if analyze:
            assert self.analyzer is not None
            tasks["analysis"] = asyncio.create_task(self.analyzer.analyze(content, context))

        # Execute
        task_items = list(tasks.items())
        task_results = await asyncio.gather(*(task for _, task in task_items))
        results_by_key = {
            name: result for (name, _), result in zip(task_items, task_results, strict=True)
        }

        logger.debug(
            "enrich done duration_ms=%.2f produced=%s",
            (time.perf_counter() - started) * 1000.0,
            sorted(results_by_key.keys()),
        )

        # Process results
        embedding = results_by_key["embedding"]
        ner_result = results_by_key.get("ner")
        entities, entity_mentions = self._resolve_entities(ner_result, text=content)

        # Merge context entities if provided
        context_entities = (context or {}).get("entities")
        if isinstance(context_entities, dict) and context_entities:
            entities = {**entities, **context_entities}

        return EnrichmentResult(
            embedding=embedding,
            summary=results_by_key.get("summary"),
            analysis=results_by_key.get("analysis"),
            keywords=results_by_key.get("keywords") or [],
            entities=entities,
            entity_mentions=entity_mentions,
        )

    async def _get_embedding(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    def _resolve_entities(
        self,
        ner_result: NERResult | None,
        *,
        text: str | None = None,
        context_window: int = 50,
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        if not ner_result:
            return {}, {}
        entities = ner_result.entities or {}
        mentions_raw = ner_result.mentions or {}
        normalized: dict[str, list[str]] = {}

        for entity_text, values in mentions_raw.items():
            normalized[entity_text] = self._resolve_entity_mentions(
                values,
                text=text,
                context_window=context_window,
            )

        return entities, normalized

    @staticmethod
    def _resolve_entity_mentions(
        values: Any,
        *,
        text: str | None,
        context_window: int,
    ) -> list[str]:
        if not isinstance(values, list):
            return []
        resolved: list[str] = []
        for value in values:
            if isinstance(value, str):
                resolved.append(value)
                continue
            if (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and isinstance(value[0], int)
                and isinstance(value[1], int)
            ):
                start, end = int(value[0]), int(value[1])
                if text:
                    context_start = max(0, start - context_window)
                    context_end = min(len(text), end + context_window)
                    resolved.append(text[context_start:context_end])
                else:
                    resolved.append(f"{start}:{end}")
        return resolved
