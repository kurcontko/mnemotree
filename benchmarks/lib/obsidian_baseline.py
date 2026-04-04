"""Obsidian-style markdown baseline memory retriever.

Simulates an Obsidian vault approach to memory:
- Each memory is a note with auto-extracted [[wiki-links]] for entities
- A link graph connects notes that share entities
- Retrieval = TF-IDF + graph boost from linked/backlinked notes
- Entities extracted via simple NER heuristics (capitalized words, dates, etc.)

This sits between flat markdown (TF-IDF only) and mnemotree (full graph + embeddings).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class _ObsidianNote:
    """A single note in the vault."""

    note_id: str
    content: str
    raw_content: str  # without wiki-link markup
    links: set[str]  # outgoing [[entity]] links
    tags: set[str]  # #tags
    timestamp: datetime


class _MemoryItemShim:
    """Matches the MemoryItem interface the benchmark harness expects."""

    def __init__(self, note: _ObsidianNote):
        self.memory_id = note.note_id
        self.content = note.raw_content
        self.memory_type = "semantic"
        self.tags = list(note.tags)
        self.entities = {e: "ENTITY" for e in note.links}
        self.importance = 0.5
        self.timestamp = note.timestamp
        self.embedding = None
        self.context: dict[str, Any] = {}


# --- Entity extraction (simple heuristics, no ML) ---

_PERSON_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{2,4})?)\b"
    r"|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
    r"(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?)\b"
    r"|\b(\d{4}[-/]\d{2}[-/]\d{2})\b",
    re.IGNORECASE,
)
_PLACE_INDICATORS = {
    "university", "hospital", "school", "college", "park", "street",
    "avenue", "city", "restaurant", "cafe", "museum", "library",
    "church", "airport", "station", "hotel", "beach", "mountain",
    "lake", "river", "building", "center", "centre", "plaza",
}


def _extract_entities(text: str) -> set[str]:
    """Extract entity-like strings from text using heuristics."""
    entities: set[str] = set()

    # Multi-word capitalized names (persons, places, orgs)
    for m in _PERSON_PATTERN.finditer(text):
        name = m.group(1)
        # Skip common sentence starters and short filler
        words = name.split()
        if len(words) >= 2 and not all(w in _SKIP_WORDS for w in words):
            entities.add(name)

    # Dates
    for m in _DATE_PATTERN.finditer(text):
        date_str = next(g for g in m.groups() if g)
        entities.add(date_str)

    # Single capitalized words that follow place indicators
    words = text.split()
    for i, w in enumerate(words):
        wl = w.lower().rstrip(".,;:!?")
        if wl in _PLACE_INDICATORS and i + 1 < len(words):
            next_w = words[i + 1].rstrip(".,;:!?")
            if next_w[0:1].isupper():
                entities.add(f"{w.rstrip('.,;:!?')} {next_w}")

    return entities


_SKIP_WORDS = {
    "The", "This", "That", "What", "When", "Where", "Which", "How",
    "And", "But", "For", "Not", "You", "Are", "Was", "Were", "Has",
    "Have", "Had", "Can", "Could", "Would", "Should", "May", "Will",
    "I", "We", "He", "She", "It", "They", "My", "Our", "His", "Her",
}


# --- Tokenization and TF-IDF (shared with markdown baseline) ---

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(documents: list[list[str]]) -> dict[str, float]:
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
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_tf: Counter[str] = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    query_tf: Counter[str] = Counter(query_tokens)

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
    return dot / norm if norm > 0 else 0.0


# --- Stats ---

@dataclass
class ObsidianBaselineStats:
    total_notes: int = 0
    total_tokens: int = 0
    total_entities: int = 0
    total_links: int = 0
    recall_count: int = 0
    total_recall_ms: float = 0.0

    @property
    def avg_recall_ms(self) -> float:
        return self.total_recall_ms / self.recall_count if self.recall_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_notes": self.total_notes,
            "total_tokens_stored": self.total_tokens,
            "total_entities": self.total_entities,
            "total_links": self.total_links,
            "recall_count": self.recall_count,
            "avg_recall_ms": round(self.avg_recall_ms, 2),
        }


# --- Main class ---

class ObsidianBaselineMemory:
    """Obsidian vault-style memory with wiki-links and graph-boosted retrieval.

    On ingestion:
    - Extracts entities from content using heuristics
    - Creates [[wiki-links]] for each entity
    - Builds a link graph: note -> entities, entity -> notes (backlinks)

    On recall:
    - TF-IDF scores all notes against query
    - Extracts entities from query
    - Boosts scores for notes linked to/from query entities (1-hop graph traversal)
    - Returns top-k by combined score

    Parameters:
        graph_boost: Weight multiplier for graph-connected notes (default 0.3).
            Score = tfidf_score + graph_boost * link_score
        backlink_depth: How many hops to traverse (1 = direct links only).
    """

    def __init__(
        self,
        graph_boost: float = 0.3,
        backlink_depth: int = 1,
    ):
        self.graph_boost = graph_boost
        self.backlink_depth = backlink_depth
        self.stats = ObsidianBaselineStats()

        self._notes: dict[str, _ObsidianNote] = {}
        self._tokenized: dict[str, list[str]] = {}
        self._idf: dict[str, float] = {}
        self._idf_dirty = True

        # Link graph
        self._entity_to_notes: dict[str, set[str]] = defaultdict(set)  # entity -> note_ids
        self._note_to_entities: dict[str, set[str]] = defaultdict(set)  # note_id -> entities

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
        """Store a note, extract entities, build links."""
        note_id = record_id or str(uuid4())

        # Extract entities
        entities = _extract_entities(content)

        # Also extract speaker name from "[date] Speaker: text" pattern
        speaker_match = re.match(r"(?:\[.*?\]\s*)?(\w+):", content)
        if speaker_match:
            speaker = speaker_match.group(1)
            if speaker[0:1].isupper() and speaker.lower() not in {"the", "a", "an", "i"}:
                entities.add(speaker)

        # Build wiki-linked version
        linked_content = content
        for entity in sorted(entities, key=len, reverse=True):
            # Only link first occurrence
            linked_content = linked_content.replace(entity, f"[[{entity}]]", 1)

        # Derive tags
        note_tags: set[str] = set(tags or [])
        if context:
            if context.get("speaker"):
                note_tags.add(f"speaker/{context['speaker']}")
            if context.get("date"):
                note_tags.add(f"date/{context['date']}")
            if context.get("session_id"):
                note_tags.add(f"session/{context['session_id']}")

        note = _ObsidianNote(
            note_id=note_id,
            content=linked_content,
            raw_content=content,
            links=entities,
            tags=note_tags,
            timestamp=datetime.now(timezone.utc),
        )
        self._notes[note_id] = note
        self._tokenized[note_id] = _tokenize(content)
        self._idf_dirty = True

        # Update link graph
        for entity in entities:
            self._entity_to_notes[entity.lower()].add(note_id)
        self._note_to_entities[note_id] = {e.lower() for e in entities}

        # Stats
        self.stats.total_notes += 1
        self.stats.total_tokens += len(content.split())
        self.stats.total_entities = len(self._entity_to_notes)
        self.stats.total_links += len(entities)

        return note_id

    def _rebuild_idf(self) -> None:
        if not self._idf_dirty:
            return
        self._idf = _build_idf(list(self._tokenized.values()))
        self._idf_dirty = False

    def _graph_scores(self, query: str) -> dict[str, float]:
        """Compute graph-based relevance scores for notes.

        1. Extract entities from query
        2. Find notes linked to those entities (backlinks)
        3. Optionally traverse 1 more hop
        4. Score = number of query entities a note is connected to / total query entities
        """
        query_entities = _extract_entities(query)
        # Also try matching query words against known entities
        query_lower = query.lower()
        for entity in self._entity_to_notes:
            if entity in query_lower:
                query_entities.add(entity)

        if not query_entities:
            return {}

        # Find connected notes (1-hop)
        note_hits: Counter[str] = Counter()
        for entity in query_entities:
            entity_lower = entity.lower()
            for note_id in self._entity_to_notes.get(entity_lower, set()):
                note_hits[note_id] += 1

        # Optional 2nd hop: notes that share entities with 1-hop notes
        if self.backlink_depth >= 2:
            hop1_notes = set(note_hits.keys())
            for nid in hop1_notes:
                for entity in self._note_to_entities.get(nid, set()):
                    for linked_nid in self._entity_to_notes.get(entity, set()):
                        if linked_nid not in hop1_notes:
                            note_hits[linked_nid] += 0.5  # weaker signal

        # Normalize by number of query entities
        max_hits = max(note_hits.values()) if note_hits else 1
        return {nid: hits / max_hits for nid, hits in note_hits.items()}

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
        """Retrieve notes using TF-IDF + graph boost."""
        start = time.perf_counter()
        limit = limit or 10

        self._rebuild_idf()
        query_tokens = _tokenize(query)

        # TF-IDF scores
        tfidf_scores: dict[str, float] = {}
        for note_id in self._notes:
            doc_tokens = self._tokenized[note_id]
            tfidf_scores[note_id] = _tfidf_score(query_tokens, doc_tokens, self._idf)

        # Graph scores
        graph_scores = self._graph_scores(query)

        # Combined score
        combined: list[tuple[float, str]] = []
        for note_id in self._notes:
            tfidf = tfidf_scores.get(note_id, 0.0)
            graph = graph_scores.get(note_id, 0.0)
            score = tfidf + self.graph_boost * graph
            combined.append((score, note_id))

        combined.sort(key=lambda x: x[0], reverse=True)
        results = [_MemoryItemShim(self._notes[nid]) for _, nid in combined[:limit]]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats.recall_count += 1
        self.stats.total_recall_ms += elapsed_ms

        return results

    async def get_embedding(self, text: str) -> list[float]:
        """Dummy embedding for interface compatibility."""
        tokens = _tokenize(text)
        vec = [0.0] * 64
        for token in tokens:
            idx = hash(token) % 64
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
