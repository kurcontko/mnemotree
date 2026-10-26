"""Reproducible source-recall evaluation; never an answer-accuracy score."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from pydantic import Field, field_validator, model_validator

from .engine import Memory
from .models import Message, Record, SearchOptions, digest, scope


class CorpusRecord(Record):
    namespace: str
    request_id: str
    session_id: str
    messages: list[Message] = Field(min_length=1)


class AnswerSpan(Record):
    source_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("source_id")
    @classmethod
    def nonblank_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer span source ID must not be blank")
        return value

    @model_validator(mode="after")
    def nonempty(self) -> AnswerSpan:
        if self.end <= self.start:
            raise ValueError("answer span end must be greater than start")
        return self


class QueryRecord(Record):
    id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_source_ids: list[str] = Field(default_factory=list)
    category: str = "unspecified"
    query_options: list[str] | None = None
    answer_spans: list[AnswerSpan] | None = None

    @field_validator("id", "namespace", "query", "category")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def valid_labels(self) -> QueryRecord:
        scope(self.namespace)
        if any(not value.strip() for value in self.relevant_source_ids):
            raise ValueError("relevant source IDs must not be blank")
        if len(set(self.relevant_source_ids)) != len(self.relevant_source_ids):
            raise ValueError("relevant source IDs must be distinct")
        if self.query_options is not None and any(
            not value.strip() for value in self.query_options
        ):
            raise ValueError("query options must not be blank")
        return self


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        # Imported from a source checkout; code_sha256 still identifies the code.
        return None


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


async def evaluate(
    memory: Memory, corpus: Path, queries: Path, options: SearchOptions, *, candidate_k: int = 100
) -> dict:
    ingestion = await ingest_corpus(memory, corpus)
    return await score_queries(memory, corpus, queries, options, ingestion, candidate_k=candidate_k)


async def ablate(memory: Memory, corpus: Path, queries: Path, options: SearchOptions) -> dict:
    if memory.extractor is None:
        raise ValueError("atomic ablation requires an extractor")
    ingestion = await ingest_corpus(memory, corpus)
    reports = {}
    for representation in ("raw", "atoms", "combined"):
        settings = SearchOptions.model_validate(
            {**options.model_dump(), "representation": representation}
        )
        reports[representation] = await score_queries(memory, corpus, queries, settings, ingestion)
    return {
        "summary": {key: report["summary"] for key, report in reports.items()},
        "shared_ingestion": ingestion,
        "reports": reports,
    }


async def ablate_context(
    memory: Memory, corpus: Path, queries: Path, options: SearchOptions
) -> dict:
    ingestion = await ingest_corpus(memory, corpus)
    reports = {}
    for name, count in (("context_off", 0), ("context_on", options.neighbor_messages or 1)):
        settings = SearchOptions.model_validate(
            {**options.model_dump(), "neighbor_messages": count}
        )
        reports[name] = await score_queries(memory, corpus, queries, settings, ingestion)
    return {
        "summary": {key: report["summary"] for key, report in reports.items()},
        "shared_ingestion": ingestion,
        "reports": reports,
    }


def retrieval_variants(memory: Memory, options: SearchOptions) -> dict[str, SearchOptions]:
    """One policy dimension at a time; every arm is validated, never model_copy."""
    variants: dict[str, dict[str, Any]] = {
        "baseline": {},
        "mode_lexical": {"mode": "lexical"},
        "context_off": {"neighbor_messages": 0},
        "context_on": {"neighbor_messages": 1},
        "role_off": {"role_first": None, "role_boost": None},
        "role_hard": {"role_first": options.role_first or "user", "role_boost": None},
        "role_soft": {"role_first": options.role_first or "user", "role_boost": 0.002},
        "model_query_combined": {"model_query_policy": "combined"},
        "model_query_question": {"model_query_policy": "question"},
        **{
            f"token_budget_{budget}": {"token_budget": budget}
            for budget in (1000, 2000, 4000, 8000)
        },
    }
    if memory.embedder is not None:
        variants.update(mode_dense={"mode": "dense"}, mode_hybrid={"mode": "hybrid"})
    elif options.mode != "lexical":
        raise ValueError("baseline dense/hybrid mode requires an embedder")
    return {
        name: SearchOptions.model_validate({**options.model_dump(), **update})
        for name, update in variants.items()
    }


async def ablate_retrieval(
    memory: Memory, corpus: Path, queries: Path, options: SearchOptions, *, candidate_k: int = 100
) -> dict:
    variants = retrieval_variants(memory, options)
    ingestion = await ingest_corpus(memory, corpus)
    reports = {}
    for name, settings in variants.items():
        reports[name] = await score_queries(
            memory, corpus, queries, settings, ingestion, candidate_k=candidate_k
        )
    return {
        "summary": {key: report["summary"] for key, report in reports.items()},
        "shared_ingestion": ingestion,
        "reports": reports,
    }


async def ingest_corpus(memory: Memory, corpus: Path) -> dict:
    # Reject contaminated runs, including forgotten-source receipts.
    if memory.store.db.execute("SELECT 1 FROM evidence_requests LIMIT 1").fetchone():
        raise ValueError("evaluation requires a fresh empty database")
    records = 0
    atom_count = 0
    source_count = 0
    started = perf_counter()
    usage_before = dict(getattr(memory.extractor, "usage", {}))
    for value in read_jsonl(corpus):
        record = CorpusRecord.model_validate(value)
        receipt = await memory.add(
            record.namespace,
            record.messages,
            request_id=record.request_id,
            session_id=record.session_id,
        )
        records += 1
        atom_count += receipt.atom_count
        source_count += len(receipt.source_ids)
    if records == 0 or source_count == 0:
        raise ValueError("evaluation requires a nonempty corpus")
    return {
        "records": records,
        "sources": source_count,
        "atoms": atom_count,
        "elapsed_ms": (perf_counter() - started) * 1000,
        "extraction_usage": {
            key: value - usage_before.get(key, 0)
            for key, value in getattr(memory.extractor, "usage", {}).items()
        },
    }


def _union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    union: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if union and start <= union[-1][1]:
            union[-1] = (union[-1][0], max(end, union[-1][1]))
        else:
            union.append((start, end))
    return union


def _answer_span_coverage(
    gold: list[AnswerSpan], retrieved: list[tuple[str, int, int]]
) -> float | None:
    if not gold:
        return None
    gold_by_source: dict[str, list[tuple[int, int]]] = {}
    for span in gold:
        gold_by_source.setdefault(span.source_id, []).append((span.start, span.end))
    retrieved_by_source: dict[str, list[tuple[int, int]]] = {}
    for source_id, start, end in retrieved:
        retrieved_by_source.setdefault(source_id, []).append((start, end))
    total = 0
    covered = 0
    for source_id, spans in gold_by_source.items():
        gold_union = _union(spans)
        retrieved_union = _union(retrieved_by_source.get(source_id, []))
        total += sum(end - start for start, end in gold_union)
        for start, end in gold_union:
            covered += sum(
                max(0, min(end, other_end) - max(start, other_start))
                for other_start, other_end in retrieved_union
            )
    return covered / total


async def score_queries(
    memory: Memory,
    corpus: Path,
    queries: Path,
    options: SearchOptions,
    ingestion: dict,
    *,
    candidate_k: int = 100,
) -> dict:
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer")
    records = [QueryRecord.model_validate(value) for value in read_jsonl(queries)]
    if not records:
        raise ValueError("evaluation requires at least one query")
    if not any(query.relevant_source_ids or query.answer_spans for query in records):
        raise ValueError("evaluation requires at least one source- or span-labeled query")
    ids = [query.id for query in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query id")
    rows = []
    for query in records:
        gold = set(query.relevant_source_ids) | {
            span.source_id for span in query.answer_spans or []
        }
        for source_id in gold:
            if memory.source(query.namespace, source_id) is None:
                raise ValueError(f"query {query.id} references an absent source: {source_id}")
        for span in query.answer_spans or []:
            source = memory.source(query.namespace, span.source_id)
            if source is None:
                raise ValueError(
                    f"query {query.id} answer span references an absent source: {span.source_id}"
                )
            if span.end > len(source.content):
                raise ValueError(
                    f"query {query.id} answer span {span.start}:{span.end} exceeds "
                    f"source {span.source_id} bounds 0:{len(source.content)}"
                )
        result = await memory.search(
            query.namespace, query.query, options=options, query_options=query.query_options
        )
        ranked = result.trace.candidates[:candidate_k]
        candidates_at_k = {item.chunk.source_id for item in ranked}
        candidates = {item.chunk.source_id for item in result.trace.candidates}
        candidate_spans = [
            (item.chunk.source_id, item.chunk.start, item.chunk.end)
            for item in result.trace.candidates
        ]
        anchor_order = list(
            dict.fromkeys(member.source_id for item in result.evidence for member in item.members())
        )
        packed_anchors = set(anchor_order)
        packed_spans = [
            (span.source_id, span.start, span.end)
            for item in result.evidence
            for member in item.members()
            for span in (member, *member.neighbors)
        ]
        packed = list(
            dict.fromkeys(source_id for item in result.evidence for source_id in item.source_ids())
        )
        expanded = candidates | {span.source_id for span in result.trace.context_candidates}
        candidate_coverage = _answer_span_coverage(query.answer_spans or [], candidate_spans)
        packed_coverage = _answer_span_coverage(query.answer_spans or [], packed_spans)
        common_spans = [
            (source_id, max(start, pstart), min(end, pend))
            for source_id, start, end in candidate_spans
            for psource, pstart, pend in packed_spans
            if source_id == psource and max(start, pstart) < min(end, pend)
        ]
        retained_coverage = _answer_span_coverage(query.answer_spans or [], common_spans)
        metrics = {
            "candidate_source_recall_at_k": len(gold & candidates_at_k) / len(gold)
            if gold
            else None,
            "candidate_reciprocal_rank_at_k": next(
                (1 / rank for rank, item in enumerate(ranked, 1) if item.chunk.source_id in gold),
                0.0,
            )
            if gold
            else None,
            "candidate_answer_span_coverage_at_k": _answer_span_coverage(
                query.answer_spans or [],
                [(item.chunk.source_id, item.chunk.start, item.chunk.end) for item in ranked],
            ),
            # Fractions of labeled gold characters, not fractions of whole sources.
            # Candidates are post-cap ranked anchors, not the full retriever pool.
            # Expansion/completion can add gold characters even while packing loses others.
            "packing_lost_answer_span_fraction": (
                candidate_coverage - retained_coverage
                if candidate_coverage is not None and retained_coverage is not None
                else None
            ),
            "packing_added_answer_span_fraction": (
                packed_coverage - retained_coverage
                if packed_coverage is not None and retained_coverage is not None
                else None
            ),
            "candidate_source_recall": len(gold & candidates) / len(gold) if gold else None,
            "expanded_source_recall": len(gold & expanded) / len(gold) if gold else None,
            "packed_anchor_source_recall": len(gold & packed_anchors) / len(gold) if gold else None,
            "packed_source_recall": len(gold & set(packed)) / len(gold) if gold else None,
            "packed_all_sources": int(gold <= set(packed)) if gold else None,
            "packed_reciprocal_rank": next(
                (1 / rank for rank, id in enumerate(packed, 1) if id in gold), 0.0
            )
            if gold
            else None,
            # Neighbors interleave with anchors in `packed`, so arms with and without
            # context are compared on anchors alone.
            "packed_anchor_reciprocal_rank": next(
                (1 / rank for rank, id in enumerate(anchor_order, 1) if id in gold), 0.0
            )
            if gold
            else None,
            "candidate_answer_span_coverage": _answer_span_coverage(
                query.answer_spans or [], candidate_spans
            ),
            "packed_answer_span_coverage": _answer_span_coverage(
                query.answer_spans or [], packed_spans
            ),
        }
        rows.append(
            {
                "id": query.id,
                "category": query.category,
                "has_gold": bool(gold),
                "has_answer_spans": bool(query.answer_spans),
                **metrics,
                "packed_source_ids": packed,
                "candidate_k": candidate_k,
                "candidate_count": len(result.trace.candidates),
                "atom_hit_count": len(result.trace.atom_hits),
                "packed_context_count": sum(
                    len(member.neighbors) for item in result.evidence for member in item.members()
                ),
                "packed_tokens": result.trace.packed_tokens,
                "timings_ms": result.trace.timings_ms,
            }
        )

    def aggregate(items):
        answerable = [row for row in items if row["has_gold"]]
        span_labeled = [row for row in items if row["has_answer_spans"]]
        return {
            "queries": len(items),
            "gold_queries": len(answerable),
            "no_gold_queries": len(items) - len(answerable),
            "answer_span_queries": len(span_labeled),
            "no_answer_span_queries": len(items) - len(span_labeled),
            **{
                key: mean(row[key] for row in answerable) if answerable else None
                for key in (
                    "candidate_source_recall_at_k",
                    "candidate_reciprocal_rank_at_k",
                    "candidate_source_recall",
                    "expanded_source_recall",
                    "packed_anchor_source_recall",
                    "packed_source_recall",
                    "packed_all_sources",
                    "packed_reciprocal_rank",
                    "packed_anchor_reciprocal_rank",
                )
            },
            **{
                key: mean(row[key] for row in span_labeled) if span_labeled else None
                for key in (
                    "candidate_answer_span_coverage",
                    "packed_answer_span_coverage",
                    "candidate_answer_span_coverage_at_k",
                    "packing_lost_answer_span_fraction",
                    "packing_added_answer_span_fraction",
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
            "corpus_records": ingestion["records"],
            "ingestion": ingestion,
            "extraction": memory.store.extractor_config,
            "mnemotree": installed_version("mnemotree"),
            "sqlite": sqlite3.sqlite_version,
            "sqlite_vec": version("sqlite-vec"),
            "tokenizer": memory.counter.signature,
            "embedding": memory.store.embedding_spec.model_dump()
            if memory.store.embedding_spec
            else None,
            "chunk_size": memory.chunk_size,
            "chunk_overlap": memory.chunk_overlap,
            "search": options.model_dump(),
            "candidate_k": candidate_k,
            "metric_notes": {
                "candidate_rank": (
                    "Top k post-source-cap anchors, after role policy and reranking; "
                    "includes repeated sources when uncapped. Excludes overflow/completion "
                    "and is not pre-cap retriever recall."
                ),
                "span_coverage": "Union of labeled source characters; overlap counted once.",
                "packing_loss": "Gold characters in post-cap ranked anchors absent from packed spans.",
                "packing_addition": "Gold characters added by windows, neighbors or completion.",
                "empty_labels": "Unknown coverage, never an unanswerable ground-truth label.",
                "source_gold": "Union of explicit relevant source IDs and answer-span source IDs.",
            },
            "reranker": None if memory.reranker is None else type(memory.reranker).__qualname__,
        },
        "summary": aggregate(rows),
        "categories": {
            category: aggregate([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "queries": rows,
    }
