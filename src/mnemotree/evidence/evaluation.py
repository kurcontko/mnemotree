"""Reproducible source-recall evaluation; never an answer-accuracy score."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from pydantic import Field

from .engine import Memory
from .models import Message, Record, SearchOptions, digest


class CorpusRecord(Record):
    namespace: str
    request_id: str
    session_id: str
    messages: list[Message] = Field(min_length=1)


class QueryRecord(Record):
    id: str
    namespace: str
    query: str
    relevant_source_ids: list[str]
    category: str = "unspecified"


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


async def evaluate(memory: Memory, corpus: Path, queries: Path, options: SearchOptions) -> dict:
    # Reject contaminated runs, including forgotten-source receipts.
    if memory.store.db.execute("SELECT 1 FROM evidence_requests LIMIT 1").fetchone():
        raise ValueError("evaluation requires a fresh empty database")
    records = 0
    for value in read_jsonl(corpus):
        record = CorpusRecord.model_validate(value)
        await memory.add(
            record.namespace,
            record.messages,
            request_id=record.request_id,
            session_id=record.session_id,
        )
        records += 1
    rows = []
    seen = set()
    for value in read_jsonl(queries):
        query = QueryRecord.model_validate(value)
        if query.id in seen:
            raise ValueError(f"duplicate query id: {query.id}")
        seen.add(query.id)
        gold = set(query.relevant_source_ids)
        for source_id in gold:
            if memory.source(query.namespace, source_id) is None:
                raise ValueError(f"query {query.id} references an absent source: {source_id}")
        result = await memory.search(query.namespace, query.query, options=options)
        candidates = {item.chunk.source_id for item in result.trace.candidates}
        packed = list(dict.fromkeys(item.source_id for item in result.evidence))
        metrics = {
            "candidate_source_recall": len(gold & candidates) / len(gold) if gold else None,
            "packed_source_recall": len(gold & set(packed)) / len(gold) if gold else None,
            "packed_all_sources": int(gold <= set(packed)) if gold else None,
            "packed_reciprocal_rank": next(
                (1 / rank for rank, id in enumerate(packed, 1) if id in gold), 0.0
            )
            if gold
            else None,
        }
        rows.append(
            {
                "id": query.id,
                "category": query.category,
                "has_gold": bool(gold),
                **metrics,
                "packed_source_ids": packed,
                "candidate_count": len(result.trace.candidates),
                "packed_tokens": result.trace.packed_tokens,
                "timings_ms": result.trace.timings_ms,
            }
        )
    if not rows:
        raise ValueError("evaluation requires at least one query")

    def aggregate(items):
        answerable = [row for row in items if row["has_gold"]]
        return {
            "queries": len(items),
            "gold_queries": len(answerable),
            "no_gold_queries": len(items) - len(answerable),
            **{
                key: mean(row[key] for row in answerable) if answerable else None
                for key in (
                    "candidate_source_recall",
                    "packed_source_recall",
                    "packed_all_sources",
                    "packed_reciprocal_rank",
                )
            },
            "mean_packed_tokens": mean(row["packed_tokens"] for row in items),
            "mean_search_ms": mean(row["timings_ms"]["total"] for row in items),
        }

    code = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "manifest": {
            "corpus_sha256": file_hash(corpus),
            "queries_sha256": file_hash(queries),
            "code_sha256": digest(code),
            "corpus_records": records,
            "mnemotree": version("mnemotree"),
            "sqlite": sqlite3.sqlite_version,
            "sqlite_vec": version("sqlite-vec"),
            "tokenizer": memory.counter.signature,
            "embedding": memory.store.embedding_spec.model_dump()
            if memory.store.embedding_spec
            else None,
            "chunk_size": memory.chunk_size,
            "chunk_overlap": memory.chunk_overlap,
            "search": options.model_dump(),
            "reranker": None if memory.reranker is None else type(memory.reranker).__qualname__,
        },
        "summary": aggregate(rows),
        "categories": {
            category: aggregate([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "queries": rows,
    }
