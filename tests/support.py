"""Deterministic stand-ins for the engine's providers, shared by snapshot tests.

None of them needs a network or a model file, and none moves with a dependency's
release, so a snapshot built on them changes only when mnemotree does.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from mnemotree.core.models import AtomDraft, Citation, EmbeddingSpec, ExtractorSpec

WORD = re.compile(r"\w+|[^\w\s]")


class WordCounter:
    """One token per word or punctuation mark, unlike tiktoken's versioned encodings."""

    signature = "golden-words"

    def count(self, text: str) -> int:
        return len(WORD.findall(text))


class HashEmbedder:
    """Deterministic bag of words: dense ranks follow the text."""

    def __init__(self, model: str = "hash", dimensions: int = 32) -> None:
        self.spec = EmbeddingSpec(provider="test", model=model, revision="1", dimensions=dimensions)

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.spec.dimensions
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % len(vector)] += 1.0 if digest[1] & 1 else -1.0
        return vector if any(vector) else [1.0] + [0.0] * (len(vector) - 1)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


class OverlapReranker:
    """Scores a text by the query words it shares, shorter texts first among equals."""

    async def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(re.findall(r"\w+", query.lower()))
        return [
            len(terms & set(re.findall(r"\w+", text.lower()))) + 1 / (1 + len(text))
            for text in texts
        ]


class OneAtomPerPassage:
    """An extractor that returns each passage as one atom quoting all of it."""

    def __init__(self, model: str = "fixture", prompt_hash: str = "test") -> None:
        self.spec = ExtractorSpec(
            provider="test", model=model, revision="1", prompt_hash=prompt_hash
        )

    async def extract(self, passages: Any) -> list[AtomDraft]:
        return [
            AtomDraft(text=p.text, citations=(Citation(passage_id=p.id, quote=p.text),))
            for p in passages
        ]
