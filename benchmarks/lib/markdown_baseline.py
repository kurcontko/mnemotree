"""Markdown-file baseline memory retriever.

Simulates how Claude Code's native .md memory works:
- Stores memories as individual .md files in a directory
- Retrieval = read all files, rank by TF-IDF similarity, return top-k
- No embeddings, no graph, no decay — just text matching

This implements the same duck-typed interface that evaluate.py expects
from MemoryCore: recall(), get_embedding(), and remember().
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class _MarkdownMemory:
    """Internal storage for a single markdown memory."""

    memory_id: str
    content: str
    memory_type: str
    tags: list[str]
    entities: dict[str, str]
    importance: float
    timestamp: datetime
    file_path: Path | None = None


class _MemoryItemShim:
    """Lightweight shim that matches the MemoryItem interface evaluate.py uses.

    evaluate.py accesses: .memory_id, .content, .embedding, .context
    """

    def __init__(self, mem: _MarkdownMemory, embedding: list[float] | None = None):
        self.memory_id = mem.memory_id
        self.content = mem.content
        self.memory_type = mem.memory_type
        self.tags = mem.tags
        self.entities = mem.entities
        self.importance = mem.importance
        self.timestamp = mem.timestamp
        self.embedding = embedding
        self.context: dict[str, Any] = {}


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across a corpus."""
    n = len(documents)
    if n == 0:
        return {}
    df: Counter[str] = Counter()
    for doc_tokens in documents:
        df.update(set(doc_tokens))
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def _tfidf_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
) -> float:
    """Compute TF-IDF cosine similarity between query and document."""
    if not query_tokens or not doc_tokens:
        return 0.0

    # Document TF
    doc_tf: Counter[str] = Counter(doc_tokens)
    doc_len = len(doc_tokens)

    # Query TF
    query_tf: Counter[str] = Counter(query_tokens)

    # Build TF-IDF vectors and compute dot product in one pass
    dot = 0.0
    query_norm_sq = 0.0
    doc_norm_sq = 0.0

    all_terms = set(query_tf) | set(doc_tf)
    for term in all_terms:
        q_tfidf = (query_tf.get(term, 0) / max(len(query_tokens), 1)) * idf.get(term, 1.0)
        d_tfidf = (doc_tf.get(term, 0) / max(doc_len, 1)) * idf.get(term, 1.0)
        dot += q_tfidf * d_tfidf
        query_norm_sq += q_tfidf * q_tfidf
        doc_norm_sq += d_tfidf * d_tfidf

    norm = math.sqrt(query_norm_sq) * math.sqrt(doc_norm_sq)
    if norm == 0:
        return 0.0
    return dot / norm


@dataclass
class MarkdownBaselineStats:
    """Token/timing statistics for the baseline retriever."""

    total_memories: int = 0
    total_tokens: int = 0  # Approximate token count (words)
    total_chars: int = 0
    recall_count: int = 0
    total_recall_ms: float = 0.0
    tokens_per_recall: list[int] = field(default_factory=list)

    @property
    def avg_recall_ms(self) -> float:
        return self.total_recall_ms / self.recall_count if self.recall_count else 0.0

    @property
    def tokens_returned_avg(self) -> float:
        return sum(self.tokens_per_recall) / len(self.tokens_per_recall) if self.tokens_per_recall else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "total_tokens_stored": self.total_tokens,
            "total_chars_stored": self.total_chars,
            "recall_count": self.recall_count,
            "avg_recall_ms": round(self.avg_recall_ms, 2),
            "avg_tokens_returned": round(self.tokens_returned_avg, 1),
        }


class MarkdownBaselineMemory:
    """Baseline memory system that simulates flat .md file storage.

    Modes:
    - "stuff": Return ALL memories (what Claude Code does with small MEMORY.md).
      Context window gets everything — no retrieval intelligence.
    - "tfidf": Rank by TF-IDF similarity, return top-k.
      Simulates what you'd get with a basic search over .md files.

    Usage:
        baseline = MarkdownBaselineMemory(mode="tfidf")
        await baseline.remember("User prefers pytest", memory_type=MemoryType.SEMANTIC, ...)
        results = await baseline.recall("test framework", limit=5)
    """

    def __init__(
        self,
        memory_dir: str | Path | None = None,
        mode: str = "tfidf",
        token_budget: int | None = None,
        embedding_dim: int = 64,
    ):
        self.memory_dir = Path(memory_dir) if memory_dir else None
        self.mode = mode
        self.token_budget = token_budget
        self.embedding_dim = embedding_dim
        self.stats = MarkdownBaselineStats()

        self._memories: dict[str, _MarkdownMemory] = {}
        self._tokenized_docs: dict[str, list[str]] = {}
        self._idf: dict[str, float] = {}
        self._idf_dirty = True

        # If a directory is provided, load existing .md files
        if self.memory_dir and self.memory_dir.exists():
            self._load_from_directory()

    def _load_from_directory(self) -> None:
        """Load all .md files from the memory directory."""
        if not self.memory_dir:
            return
        for md_file in sorted(self.memory_dir.glob("**/*.md")):
            content = md_file.read_text().strip()
            if not content:
                continue
            mem_id = str(uuid4())
            mem = _MarkdownMemory(
                memory_id=mem_id,
                content=content,
                memory_type="semantic",
                tags=[],
                entities={},
                importance=0.5,
                timestamp=datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc),
                file_path=md_file,
            )
            self._memories[mem_id] = mem
            self._tokenized_docs[mem_id] = _tokenize(content)
            self._idf_dirty = True

    async def remember(
        self,
        content: str,
        memory_type: Any = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        context: dict[str, Any] | None = None,
        analyze: bool = False,
        summarize: bool = False,
        record_id: str | None = None,
    ) -> str:
        """Store a memory. Mirrors MemoryCore.remember() signature."""
        mem_id = record_id or str(uuid4())
        type_str = memory_type.value if hasattr(memory_type, "value") else str(memory_type or "semantic")

        entities: dict[str, str] = {}
        if context and "entities" in context:
            entities = context["entities"]

        mem = _MarkdownMemory(
            memory_id=mem_id,
            content=content,
            memory_type=type_str,
            tags=tags or [],
            entities=entities,
            importance=importance or 0.5,
            timestamp=datetime.now(timezone.utc),
        )
        self._memories[mem_id] = mem
        self._tokenized_docs[mem_id] = _tokenize(content)
        self._idf_dirty = True

        # Update stats
        tokens = len(content.split())
        self.stats.total_memories += 1
        self.stats.total_tokens += tokens
        self.stats.total_chars += len(content)

        # Optionally persist to disk
        if self.memory_dir:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.memory_dir / f"{mem_id}.md"
            file_path.write_text(content)
            mem.file_path = file_path

        return mem_id

    def _rebuild_idf(self) -> None:
        """Rebuild IDF index from all stored documents."""
        if not self._idf_dirty:
            return
        all_docs = list(self._tokenized_docs.values())
        self._idf = _build_idf(all_docs)
        self._idf_dirty = False

    async def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        scoring: bool | None = None,
        update_access: bool | None = None,
        filters: Any = None,
        candidate_limit: int | None = None,
        options: Any = None,
    ) -> list[_MemoryItemShim]:
        """Retrieve memories matching a query.

        In 'stuff' mode: returns all memories (truncated to token_budget).
        In 'tfidf' mode: ranks by TF-IDF cosine similarity, returns top-k.
        """
        start = time.perf_counter()
        limit = limit or 10

        if self.mode == "stuff":
            results = self._recall_stuff(limit)
        else:
            results = self._recall_tfidf(query, limit)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats.recall_count += 1
        self.stats.total_recall_ms += elapsed_ms

        # Track tokens returned
        returned_tokens = sum(len(r.content.split()) for r in results)
        self.stats.tokens_per_recall.append(returned_tokens)

        return results

    def _recall_stuff(self, limit: int) -> list[_MemoryItemShim]:
        """Return all memories up to token budget (context stuffing)."""
        all_mems = sorted(self._memories.values(), key=lambda m: m.timestamp)
        results = []
        total_tokens = 0

        for mem in all_mems:
            tokens = len(mem.content.split())
            if self.token_budget and total_tokens + tokens > self.token_budget:
                break
            results.append(_MemoryItemShim(mem))
            total_tokens += tokens

        return results[:limit] if limit else results

    def _recall_tfidf(self, query: str, limit: int) -> list[_MemoryItemShim]:
        """Rank memories by TF-IDF similarity and return top-k."""
        self._rebuild_idf()
        query_tokens = _tokenize(query)

        scored: list[tuple[float, _MarkdownMemory]] = []
        for mem_id, mem in self._memories.items():
            doc_tokens = self._tokenized_docs[mem_id]
            score = _tfidf_score(query_tokens, doc_tokens, self._idf)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [_MemoryItemShim(mem) for _, mem in scored[:limit]]

    async def get_embedding(self, text: str) -> list[float]:
        """Return a deterministic bag-of-words embedding (same as DummyEmbeddings)."""
        tokens = _tokenize(text)
        vec = [0.0] * self.embedding_dim
        for token in tokens:
            idx = hash(token) % self.embedding_dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
