from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import spacy


class KeywordExtractor(Protocol):
    async def extract(self, text: str) -> list[str]: ...


@dataclass
class SpacyKeywordExtractor:
    model: str = "en_core_web_sm"
    max_keywords: int = 8
    min_length: int = 3

    def __post_init__(self) -> None:
        self.nlp = spacy.load(self.model, disable=["ner", "textcat"])

    async def extract(self, text: str) -> list[str]:
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> list[str]:
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and len(token.text) >= self.min_length
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        counts = Counter(tokens)
        return [token for token, _ in counts.most_common(self.max_keywords)]


@dataclass
class YakeKeywordExtractor:
    language: str = "en"
    max_keywords: int = 8
    ngram_size: int = 3
    dedup_limit: float = 0.9
    dedup_func: str = "seqm"
    window_size: int = 1
    min_length: int = 3
    stopwords: list[str] | None = None

    def __post_init__(self) -> None:
        try:
            import yake
        except ImportError as exc:
            raise RuntimeError(
                "yake is required for YakeKeywordExtractor. "
                "Install mnemotree with the 'keywords' extra or add the package."
            ) from exc
        self._extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.ngram_size,
            dedupLim=self.dedup_limit,
            dedupFunc=self.dedup_func,
            windowsSize=self.window_size,
            top=self.max_keywords,
            stopwords=self.stopwords,
        )

    async def extract(self, text: str) -> list[str]:
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> list[str]:
        keywords = self._extractor.extract_keywords(text)
        results = [kw for kw, _ in keywords]
        if self.min_length <= 1:
            return results
        return [kw for kw in results if len(kw) >= self.min_length]
