"""Golden retrieval snapshots: refactors must leave every observable result unchanged.

Each case runs a fixed corpus through a fixed option set with deterministic providers
(hash embedder, overlap reranker, one atom per passage, a word tokenizer), so the
snapshot needs no network and does not move with tiktoken releases. Only wall-clock
timings and ingestion times are removed.

A deliberate retrieval change regenerates the file, and its diff is the review:

    MNEMOTREE_UPDATE_GOLDEN=1 pytest tests/test_golden_retrieval.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.evidence.atoms import AtomDraft, Citation, ExtractorSpec
from mnemotree.evidence.models import EmbeddingSpec

GOLDEN = Path(__file__).parent / "golden" / "retrieval"
UPDATE = os.environ.get("MNEMOTREE_UPDATE_GOLDEN") == "1"
WORD = re.compile(r"\w+|[^\w\s]")
HEX_ID = re.compile(r"\b([0-9a-f]{12})[0-9a-f]{52}\b")
START = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)

DEPLOY_LOG = " ".join(
    f"Step {index}: the migration of shard {index} finished with {index % 4} warnings "
    f"and the deployment checks for region {'eu' if index % 2 else 'us'} passed."
    for index in range(1, 13)
)
SESSIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "garden",
        [
            ("user", "I planted tomatoes and basil in the raised garden bed yesterday."),
            ("assistant", "Tomatoes and basil grow well together; water them in the morning."),
            ("user", "The basil already has slugs on it."),
            ("assistant", "Copper tape around the bed keeps slugs away."),
            ("user", "Next week I will add marigolds along the garden fence."),
        ],
    ),
    (
        "travel",
        [
            ("user", "The conference is in Lisbon this year, not Porto."),
            ("assistant", "Lisbon in May is warm; book the hotel near the venue early."),
            ("user", "My flight to Lisbon leaves on Friday at 7am."),
            ("assistant", "Leave for the airport by 5am for an international flight."),
        ],
    ),
    (
        "ops",
        [
            ("user", "Can you summarise the deployment migration warnings?"),
            ("assistant", DEPLOY_LOG),
            ("user", "Which shard had the most warnings during the migration?"),
            ("assistant", "Shards 3, 7 and 11 each reported 3 warnings."),
            ("user", "数据库迁移在周二完成。"),
        ],
    ),
]
QUERIES: list[tuple[str, list[str] | None]] = [
    ("what did I plant in the garden", None),
    ("Which city is the conference in?", ["A. Lisbon", "B. Porto"]),
    ("deployment migration warnings for shard 7", None),
]


class WordCounter:
    signature = "golden-words"

    def count(self, text: str) -> int:
        return len(WORD.findall(text))


class HashEmbedder:
    spec = EmbeddingSpec(provider="test", model="golden-hash", revision="1", dimensions=16)

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
    async def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(re.findall(r"\w+", query.lower()))
        return [
            len(terms & set(re.findall(r"\w+", text.lower()))) + 1 / (1 + len(text))
            for text in texts
        ]


class OneAtomPerPassage:
    spec = ExtractorSpec(provider="test", model="golden", revision="1", prompt_hash="golden")

    async def extract(self, passages: Any) -> list[AtomDraft]:
        return [
            AtomDraft(text=p.text, citations=(Citation(passage_id=p.id, quote=p.text),))
            for p in passages
        ]


def aml_preset() -> dict[str, Any]:
    pytest.importorskip("fastapi")
    from mnemotree.evidence.server import aml_options

    class Embedded:
        embedder = True

    # Bounded so the snapshot stays reviewable; every policy of the preset is kept.
    preset = aml_options(Embedded())  # type: ignore[arg-type]
    return {**preset.model_dump(), "limit": 6, "token_budget": 400}


# name -> (corpus kind, search options). "lexical" has no embedder, "hybrid" a hash
# embedder, "reranked" adds the overlap reranker, "atoms" adds the extractor.
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "lexical": ("lexical", {"mode": "lexical"}),
    "lexical-small-budget": ("lexical", {"mode": "lexical", "token_budget": 40}),
    "dense": ("hybrid", {"mode": "dense"}),
    "hybrid": ("hybrid", {"mode": "hybrid"}),
    "hybrid-weights": ("hybrid", {"mode": "hybrid", "lexical_weight": 2.0, "rrf_k": 10}),
    "reranked": ("reranked", {"mode": "hybrid"}),
    "neighbor-chunks": ("hybrid", {"mode": "hybrid", "neighbor_chunks": 1}),
    "neighbor-messages": ("hybrid", {"mode": "hybrid", "neighbor_messages": 1}),
    "merge-blocks": ("hybrid", {"mode": "hybrid", "neighbor_messages": 1, "merge_blocks": True}),
    "order-time": ("hybrid", {"mode": "hybrid", "order": "time", "neighbor_messages": 1}),
    "chunks-per-source": ("hybrid", {"mode": "hybrid", "chunks_per_source": 1}),
    "role-first": ("reranked", {"mode": "hybrid", "role_first": "user"}),
    "role-boost": ("reranked", {"mode": "hybrid", "role_first": "user", "role_boost": 0.01}),
    "role-gate-closed": (
        "hybrid",
        {"mode": "hybrid", "role_first": "user", "role_first_min_assistant_share": 0.99},
    ),
    "deep-pool": (
        "hybrid",
        {"mode": "hybrid", "candidate_limit": 4, "deep_candidate_limit": 30, "limit": 4},
    ),
    "question-policy": ("hybrid", {"mode": "hybrid", "model_query_policy": "question"}),
    "atoms": ("atoms", {"mode": "hybrid", "representation": "atoms"}),
    "combined": ("atoms", {"mode": "hybrid", "representation": "combined", "atom_weight": 0.5}),
    "aml-preset": ("hybrid", {}),
}


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: scrub(item)
            for key, item in value.items()
            if key not in ("timings_ms", "ingested_at")
        }
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        # Chunk and atom IDs are SHA-256 digests; a prefix is unique in this corpus.
        return HEX_ID.sub(r"\1", value)
    if isinstance(value, float):
        # BM25 and float32 cosine scores can differ in the last bits between the
        # SQLite builds CI runs on; ranks and order stay exact.
        return float(f"{value:.9g}")
    return value


def fingerprint(value: Any) -> Any:
    """Replace repeated text with a digest; offsets and IDs still pin it down."""
    if isinstance(value, dict):
        return {
            key: short(item) if key in ("text", "content") else fingerprint(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [fingerprint(item) for item in value]
    return value


def short(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()[:12]} len={len(text)}"


def snapshot(dump: dict[str, Any]) -> dict[str, Any]:
    # The packed output stays verbatim so a reviewer reads what changed; candidates,
    # sources and atoms repeat corpus text and keep only a digest of it.
    trace = dump["trace"]
    for key in ("candidates", "atom_hits"):
        trace[key] = fingerprint(trace[key])
    dump["sources"] = fingerprint(dump["sources"])
    return scrub(dump)


async def run_case(tmp_path: Path, kind: str, options: dict[str, Any]) -> list[Any]:
    memory = Memory(
        tmp_path / "golden.db",
        embedder=HashEmbedder() if kind != "lexical" else None,
        reranker=OverlapReranker() if kind == "reranked" else None,
        extractor=OneAtomPerPassage() if kind == "atoms" else None,
        token_counter=WordCounter(),
        chunk_size=200,
        chunk_overlap=40,
        extraction_batch_chars=400,
    )
    with memory:
        stamp = START
        for session, turns in SESSIONS:
            messages = []
            for index, (role, content) in enumerate(turns):
                stamp += timedelta(minutes=7)
                messages.append(
                    Message(id=f"{session}-{index}", role=role, content=content, timestamp=stamp)
                )
            await memory.add("golden", messages, request_id=session, session_id=session)
            stamp += timedelta(days=1)
        search = SearchOptions.model_validate(
            {"candidate_limit": 10, "limit": 6, "token_budget": 400, **options}
            if options
            else aml_preset()
        )
        results = []
        for query, answers in QUERIES:
            result = await memory.search("golden", query, options=search, query_options=answers)
            results.append(
                snapshot(
                    {
                        **result.model_dump(mode="json"),
                        "sources": [source.model_dump(mode="json") for source in result.sources],
                    }
                )
            )
        return results


@pytest.mark.parametrize("name", sorted(CASES))
async def test_retrieval_matches_golden_snapshot(tmp_path, name):
    kind, options = CASES[name]
    actual = await run_case(tmp_path, kind, options)
    assert any(result["evidence"] for result in actual), "case retrieves nothing"
    path = GOLDEN / f"{name}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    assert path.exists(), "no snapshot: run with MNEMOTREE_UPDATE_GOLDEN=1"
    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_every_snapshot_has_a_case():
    assert {path.stem for path in GOLDEN.glob("*.json")} == set(CASES)
