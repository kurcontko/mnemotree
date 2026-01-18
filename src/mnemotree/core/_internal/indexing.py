from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..models import MemoryItem


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into a list of alphanumeric tokens.

    Args:
        text: Input string to tokenize

    Returns:
        List of lowercase alphanumeric tokens
    """
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


PRF_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@runtime_checkable
class LexicalIndex(Protocol):
    """Protocol for a lexical search index."""

    def add(self, memory: MemoryItem) -> None: ...
    def remove(self, memory_id: str) -> None: ...
    def search(self, query: str, k: int) -> list[str] | list[tuple[str, float]]: ...


@runtime_checkable
class QueryExpander(Protocol):
    """Protocol for query expansion."""

    def expand(self, tokens: list[str], top_k_docs: list[tuple[str, float]]) -> list[str]: ...


@dataclass
class BM25Index:
    """
    In-memory BM25 inverted index implementation.

    Warning:
        This index stores term frequencies for every document in memory.
        It is not optimized for large datasets (millions of documents).
        For large scale use cases, consider using an external search engine.
    """

    k1: float = 1.2
    b: float = 0.75
    doc_freq: Counter[str] = field(default_factory=Counter)
    term_freqs: dict[str, Counter[str]] = field(default_factory=dict)
    doc_len: dict[str, int] = field(default_factory=dict)
    total_len: int = 0

    def add(self, memory_id: str, tokens: list[str]) -> None:
        if not tokens:
            self.term_freqs[memory_id] = Counter()
            self.doc_len[memory_id] = 0
            return

        if memory_id in self.term_freqs:
            self._remove(memory_id)

        term_freq = Counter(tokens)
        self.term_freqs[memory_id] = term_freq
        length = sum(term_freq.values())
        self.doc_len[memory_id] = length
        self.total_len += length
        for term in term_freq:
            self.doc_freq[term] += 1

    def _remove(self, memory_id: str) -> None:
        existing = self.term_freqs.get(memory_id)
        if not existing:
            return
        for term in existing:
            self.doc_freq[term] -= 1
            if self.doc_freq[term] <= 0:
                del self.doc_freq[term]
        length = self.doc_len.get(memory_id, 0)
        self.total_len = max(0, self.total_len - length)
        self.term_freqs.pop(memory_id, None)
        self.doc_len.pop(memory_id, None)

    def remove(self, memory_id: str) -> None:
        self._remove(memory_id)

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        avgdl, n_docs = self._resolve_search_stats(query_tokens)
        if avgdl <= 0 or n_docs <= 0:
            return []

        query_terms = Counter(query_tokens)
        scores: dict[str, float] = {}

        for term in query_terms:
            idf = self._term_idf(term, n_docs)
            if idf is None:
                continue
            self._accumulate_term_scores(term, idf, avgdl, scores)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _resolve_search_stats(self, query_tokens: list[str]) -> tuple[float, int]:
        if not query_tokens or not self.term_freqs:
            return 0.0, 0
        n_docs = len(self.term_freqs)
        avgdl = self.total_len / n_docs if n_docs else 0.0
        return avgdl, n_docs

    def _term_idf(self, term: str, n_docs: int) -> float | None:
        df = self.doc_freq.get(term, 0)
        if df <= 0:
            return None
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _accumulate_term_scores(
        self, term: str, idf: float, avgdl: float, scores: dict[str, float]
    ) -> None:
        for memory_id, tf_counter in self.term_freqs.items():
            tf = tf_counter.get(term, 0)
            if tf <= 0:
                continue
            dl = self.doc_len.get(memory_id, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            score = idf * (tf * (self.k1 + 1)) / denom if denom else 0.0
            scores[memory_id] = scores.get(memory_id, 0.0) + score


class BaseQueryExpander:
    """Implements Pseudo-Relevance Feedback (PRF) for query expansion."""

    def __init__(self, index: BM25Index, top_docs: int = 5, top_terms: int = 8):
        self.index = index
        self.top_docs = top_docs
        self.top_terms = top_terms

    def expand(self, tokens: list[str], top_k_docs: list[tuple[str, float]]) -> list[str]:
        if not self._can_expand(tokens, top_k_docs):
            return tokens

        content_query_terms = self._filter_content_terms(tokens)
        if not self._has_index_terms(content_query_terms):
            return tokens

        n_docs = len(self.index.term_freqs)
        if n_docs <= 0:
            return tokens

        term_scores = self._collect_prf_scores(
            tokens=tokens,
            top_k_docs=top_k_docs,
            n_docs=n_docs,
        )
        if not term_scores:
            return tokens

        expansion_terms = self._select_expansion_terms(term_scores)
        if not expansion_terms:
            return tokens
        return list(dict.fromkeys(tokens + expansion_terms))

    def _can_expand(self, tokens: list[str], top_k_docs: list[tuple[str, float]]) -> bool:
        return bool(
            self.index and top_k_docs and self.top_docs > 0 and self.top_terms > 0 and tokens
        )

    @staticmethod
    def _filter_content_terms(tokens: list[str]) -> list[str]:
        return [
            term
            for term in tokens
            if len(term) >= 4 and term not in PRF_STOPWORDS and not term.isdigit()
        ]

    def _has_index_terms(self, content_terms: list[str]) -> bool:
        if not content_terms:
            return False
        return any(self.index.doc_freq.get(term, 0) > 0 for term in content_terms)

    def _collect_prf_scores(
        self,
        *,
        tokens: list[str],
        top_k_docs: list[tuple[str, float]],
        n_docs: int,
    ) -> dict[str, float]:
        top_doc_ids = [mid for mid, _ in top_k_docs[: self.top_docs]]
        query_term_set = set(tokens)
        term_scores: dict[str, float] = {}

        for memory_id in top_doc_ids:
            self._score_terms_from_doc(memory_id, query_term_set, n_docs, term_scores)

        return term_scores

    def _score_terms_from_doc(
        self,
        memory_id: str,
        query_term_set: set[str],
        n_docs: int,
        term_scores: dict[str, float],
    ) -> None:
        """Extract and score terms from a single document for PRF."""
        tf_counter = self.index.term_freqs.get(memory_id)
        if not tf_counter:
            return

        dl = self.index.doc_len.get(memory_id, 0)
        if dl <= 0:
            return

        for term, tf in tf_counter.items():
            if self._should_skip_term(term, tf, query_term_set):
                continue

            df = self.index.doc_freq.get(term, 0)
            if df <= 0:
                continue

            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            term_scores[term] = term_scores.get(term, 0.0) + idf * (tf / dl)

    def _should_skip_term(self, term: str, tf: int, query_term_set: set[str]) -> bool:
        """Check if term should be skipped for PRF expansion."""
        return (
            tf <= 0
            or term in query_term_set
            or term in PRF_STOPWORDS
            or len(term) < 4
            or term.isdigit()
        )

    def _select_expansion_terms(self, term_scores: dict[str, float]) -> list[str]:
        return [
            term
            for term, _ in sorted(term_scores.items(), key=lambda item: item[1], reverse=True)[
                : self.top_terms
            ]
        ]


class IndexManager:
    """Manages the lifecycle and searching of the lexical index."""

    def __init__(
        self,
        enable_bm25: bool = False,
        enable_prf: bool = False,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        prf_docs: int = 5,
        prf_terms: int = 8,
    ):
        self.enable_bm25 = enable_bm25
        self.enable_prf = enable_prf
        self.rrf_k = rrf_k
        self._index: BM25Index | None = BM25Index(k1=bm25_k1, b=bm25_b) if enable_bm25 else None
        self._cache: dict[str, MemoryItem] = {}
        self._expander: BaseQueryExpander | None = (
            BaseQueryExpander(self._index, prf_docs, prf_terms)
            if enable_bm25 and enable_prf and self._index
            else None
        )

    def add(self, memory: MemoryItem) -> None:
        if not self.enable_bm25 or not self._index:
            return

        self._cache[memory.memory_id] = memory
        tokens = tokenize(memory.content)
        if memory.tags:
            for tag in memory.tags:
                tokens.extend(tokenize(tag))
        self._index.add(memory.memory_id, tokens)

    def remove(self, memory_id: str) -> None:
        if not self.enable_bm25 or not self._index:
            return
        self._cache.pop(memory_id, None)
        self._index.remove(memory_id)

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        return self._cache.get(memory_id)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.enable_bm25 or not self._index:
            return []

        query_tokens = tokenize(query)
        ranked = self._index.search(query_tokens, top_k=k)

        if self.enable_prf and self._expander:
            expanded_tokens = self._expander.expand(query_tokens, ranked)
            if expanded_tokens != query_tokens:
                return self._search_with_expansion(
                    query_tokens=query_tokens,
                    initial_ranked=ranked,
                    k=k,
                )
        return ranked

    def _search_with_expansion(
        self, query_tokens: list[str], initial_ranked: list[tuple[str, float]], k: int
    ) -> list[tuple[str, float]]:
        if not self._index or not self._expander:
            return initial_ranked

        expanded_tokens = self._expander.expand(query_tokens, initial_ranked)
        if expanded_tokens == query_tokens:
            return initial_ranked

        ranked_prf = self._index.search(expanded_tokens, top_k=k)
        return self._merge_ranked(
            ranked_lists=[initial_ranked, ranked_prf],
            rrf_k=self.rrf_k,
            weights=[0.7, 0.3],
            k=k,
        )

    def _merge_ranked(
        self,
        *,
        ranked_lists: list[list[tuple[str, float]]],
        rrf_k: int,
        weights: list[float] | None = None,
        k: int,
    ) -> list[tuple[str, float]]:
        if not ranked_lists:
            return []
        if weights is None:
            weights = [1.0 for _ in ranked_lists]
        if len(weights) != len(ranked_lists):
            weights = [1.0 for _ in ranked_lists]

        scores: dict[str, float] = {}
        for stage_weight, ranked in zip(weights, ranked_lists, strict=True):
            if stage_weight <= 0:
                continue
            for rank, (memory_id, _) in enumerate(ranked, start=1):
                scores[memory_id] = scores.get(memory_id, 0.0) + stage_weight / (rrf_k + rank)

        merged = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return merged[:k]

    @property
    def doc_count(self) -> int:
        return len(self._cache)

    def on_remember(self, memory: MemoryItem) -> None:
        self.add(memory)

    def on_forget(self, memory_id: str) -> None:
        self.remove(memory_id)

    def bm25_candidates(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search(query, k)
