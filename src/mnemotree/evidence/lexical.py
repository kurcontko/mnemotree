"""Lexical terms shared by indexing and queries.

FTS5 ``unicode61`` keeps an unspaced CJK run as one token, so ``华沙`` never
matches ``我搬到了华沙``. Such runs are indexed as overlapping character bigrams;
other words pass through unchanged for FTS5 to normalize and Porter-stem.
Queries drop English stopwords: OR-ing "the" and "did" into BM25 lets frequent
function words outrank the content terms.

Each row also carries its namespace as a digits-only token, so a query matches only
its own namespace's postings instead of scoring the whole corpus and filtering after.
Digits pass the Porter stemmer unchanged, so distinct namespaces cannot stem together.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from collections.abc import Sequence

# Index and query must share one tokenizer; changing it requires a rebuild.
FTS_TOKENIZER = "porter unicode61"
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
# Han (including extension A and compatibility ideographs) and kana.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


def tokens(text: str) -> list[str]:
    result: list[str] = []
    for word in _WORD.findall(text):
        last = 0
        for run in _CJK.finditer(word):
            if run.start() > last:
                result.append(word[last : run.start()])
            chars = run.group()
            result.extend(
                [chars] if len(chars) == 1 else [chars[i : i + 2] for i in range(len(chars) - 1)]
            )
            last = run.end()
        if last < len(word):
            result.append(word[last:])
    return result


def index_text(text: str) -> str:
    return " ".join(tokens(text))


def scope_token(namespace: str) -> str:
    """A namespace as one index token; queries still join on the exact namespace."""
    return "n" + str(int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:16], "big"))


def scoped_match(namespace: str, query: str, *, options: Sequence[str] | None = None) -> str | None:
    """``match_expression`` restricted to rows indexed under ``namespace``."""
    match = match_expression(query, options=options)
    if match is None:
        return None
    return f'scope : "{scope_token(namespace)}" AND text : ({match})'


def match_expression(
    query: str, *, max_terms: int = 64, options: Sequence[str] | None = None
) -> str | None:
    """OR of distinct terms, balancing long queries and explicit answer options.

    Reserve half the slots for the question and share the rest among options.
    Unused slots are redistributed, and each overlong part supplies both its head
    and tail. Explicit option boundaries come from the caller, never heuristics.
    Short queries retain their original term order and stopword-only fallback.
    """
    limit = min(64, max(0, max_terms))
    parts = [query, *(options or ())]
    raw = [list(dict.fromkeys(term.lower() for term in tokens(part))) for part in parts]
    lanes = [[term for term in lane if term not in STOPWORDS] for lane in raw]
    # Preserve the historical fallback only when the entire request is stopwords.
    if not any(lanes):
        lanes = raw
    queues: list[deque[str]] = []
    for lane in lanes:
        # Alternating ends prevents a long transcript or option losing its tail.
        priority = []
        left, right = 0, len(lane) - 1
        while left <= right:
            priority.append(lane[left])
            if left < right:
                priority.append(lane[right])
            left += 1
            right -= 1
        queues.append(deque(priority))
    chosen: set[str] = set()

    def take(lane: int) -> bool:
        while queues[lane]:
            term = queues[lane].popleft()
            if term not in chosen:
                chosen.add(term)
                return True
        return False

    if len(lanes) > 1:
        for _ in range((limit + 1) // 2):
            if not take(0):
                break
        # Round-robin gives every option equal opportunity, including when one
        # option repeats question terms or has fewer terms than its allocation.
        while len(chosen) < limit:
            added = False
            for lane_index in range(1, len(lanes)):
                if len(chosen) == limit:
                    break
                added = take(lane_index) or added
            if not added:
                break
    # Fill spare option/question capacity without repeating any selected term.
    while len(chosen) < limit:
        added = False
        for lane_index in range(len(lanes)):
            if len(chosen) == limit:
                break
            added = take(lane_index) or added
        if not added:
            break
    # Restore source order within each part. OR order is immaterial to FTS5,
    # but retaining it makes ordinary short queries byte-for-byte compatible.
    terms = list(dict.fromkeys(term for lane in lanes for term in lane if term in chosen))
    if not terms:
        return None
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
