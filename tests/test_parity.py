"""Output parity for the text tracks: the service preset must keep returning the same bytes.

`test_golden_retrieval.py` pins each retrieval option on a small corpus with bounded
options. This suite pins what a deployment actually serves: `aml_options()` exactly as
shipped (the resolved options are part of every snapshot, so changing any default
fails here), on histories shaped like LoCoMo and LongMemEval v1, through
`engine.search` and through the Add/Search HTTP contract.

Evidence IDs and order, rendered content, candidate ranks, channel counts and token
counts are compared exactly; scores to 6 decimals, because sqlite-vec computes in
float32; timings not at all. Repeated text is stored as a digest and a length, except
in the service snapshot, which keeps what the platform's answer model would read.

A change that is meant to alter text-track output regenerates the files, and their
diff is the review:

    MNEMOTREE_UPDATE_PARITY=1 pytest tests/test_parity.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mnemotree.core.models import AtomDraft, Citation, SearchOptions, SearchResult
from mnemotree.evidence import Memory, Message
from mnemotree.service.presets import aml_options
from support import HashEmbedder, OneAtomPerPassage, WordCounter

PARITY = Path(__file__).parent / "parity"
SNAPSHOTS = PARITY / "snapshots"
CORPUS = json.loads((PARITY / "fixtures" / "corpus.json").read_text(encoding="utf-8"))
UPDATE = os.environ.get("MNEMOTREE_UPDATE_PARITY") == "1"
HEX_ID = re.compile(r"\b([0-9a-f]{12})[0-9a-f]{52}\b")
SERVICE = "service"
ATOM_CHARS = 160

# name -> (providers, overrides of the service preset, whether candidates are listed).
# "lexical" has no embedder, "hybrid" the hash embedder, "atoms" adds the extractor.
# Variants that only repackage the "aml" candidates keep a digest of them.
CASES: dict[str, tuple[str, dict[str, Any], bool]] = {
    "aml": ("hybrid", {}, True),
    "aml-lexical": ("lexical", {}, True),
    "order-time": ("hybrid", {"order": "time"}, False),
    "merge-off": ("hybrid", {"merge_blocks": False}, False),
    "neighbors-0": ("hybrid", {"neighbor_messages": 0}, False),
    "neighbors-2": ("hybrid", {"neighbor_messages": 2}, False),
    "small-budget": ("hybrid", {"token_budget": 600}, False),
    "atoms": ("atoms", {"representation": "atoms"}, True),
    "combined": ("atoms", {"representation": "combined"}, True),
}


class LeadAtoms(OneAtomPerPassage):
    """One atom per passage, quoting its opening: long turns exceed an atom's length."""

    async def extract(self, passages: Any) -> list[AtomDraft]:
        return [
            AtomDraft(
                text=p.text[:ATOM_CHARS],
                citations=(Citation(passage_id=p.id, quote=p.text[:ATOM_CHARS]),),
            )
            for p in passages
        ]


def short(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()[:12]} len={len(text)}"


def ident(value: str) -> str:
    # Chunk and atom IDs are SHA-256 digests; a prefix is unique in this corpus.
    return HEX_ID.sub(r"\1", value)


def score(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, str):
        return ident(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def open_memory(path: Path, kind: str) -> Memory:
    # Production chunking and extraction batching; only the providers are stand-ins.
    return Memory(
        path,
        embedder=HashEmbedder("parity-hash", 32) if kind != "lexical" else None,
        extractor=LeadAtoms("parity", "parity") if kind == "atoms" else None,
        token_counter=WordCounter(),
    )


def timestamp(message: dict[str, Any]) -> datetime | None:
    value = message.get("timestamp")
    return None if value is None else datetime.fromtimestamp(value / 1000, tz=UTC)


async def ingest(memory: Memory) -> None:
    """Add every session the way the service does: one request each, blanks skipped."""
    for user in CORPUS["users"]:
        for session in user["sessions"]:
            messages = [
                Message(content=m["content"], role=m["role"], timestamp=timestamp(m))
                for m in session["messages"]
                if m["content"].strip()
            ]
            await memory.add(
                user["user_id"],
                messages,
                request_id=session["session_id"],
                session_id=session["session_id"],
            )


def describe(result: SearchResult, *, candidates: bool) -> dict[str, Any]:
    trace = result.trace
    rows = [
        " ".join(
            [
                ident(candidate.chunk.id),
                f"{candidate.fusion_score:.6f}",
                *(f"{name}={channel.rank}" for name, channel in candidate.channels.items()),
            ]
        )
        for candidate in trace.candidates
    ]
    atom_hits = canonical(json.loads(json.dumps(trace.atom_hits, default=str)))
    return {
        "context": short(result.context),
        "evidence": [
            {
                "id": ident(item.id),
                "source_id": ident(item.source_id),
                "span": [item.start, item.end],
                "score": score(item.score),
                "rerank_score": score(item.rerank_score),
                "merged": [ident(member.id) for member in item.merged],
                "neighbors": [
                    ident(span.id) for member in item.members() for span in member.neighbors
                ],
                "content": short(item.render()),
            }
            for item in result.evidence
        ],
        "candidates": rows if candidates else short("\n".join(rows)),
        "atom_hits": [short(json.dumps(hit, sort_keys=True)) for hit in atom_hits],
        "trace": canonical(
            trace.model_dump(
                mode="json",
                exclude={"candidates", "atom_hits", "context_candidates", "timings_ms"},
            )
        )
        | {"context_candidates": [ident(span.id) for span in trace.context_candidates]},
    }


async def run_engine(tmp_path: Path, name: str) -> dict[str, Any]:
    kind, overrides, candidates = CASES[name]
    with open_memory(tmp_path / "parity.db", kind) as memory:
        await ingest(memory)
        options = SearchOptions.model_validate({**aml_options(memory).model_dump(), **overrides})
        results = []
        for user in CORPUS["users"]:
            for question in user["questions"]:
                result = await memory.search(
                    user["user_id"],
                    question["query"],
                    options=options,
                    query_options=question.get("options"),
                )
                results.append(
                    {
                        "user_id": user["user_id"],
                        "query": question["query"],
                        "options": question.get("options"),
                        **describe(result, candidates=candidates),
                    }
                )
        return {"search_options": options.model_dump(mode="json"), "results": results}


async def run_service(tmp_path: Path) -> dict[str, Any]:
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from mnemotree.service.app import create_app

    with open_memory(tmp_path / "parity.db", "hybrid") as memory:
        app = create_app(memory, api_key="parity-key")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer parity-key"},
        ) as client:
            for user in CORPUS["users"]:
                for session in user["sessions"]:
                    payload = {
                        "request_id": session["session_id"],
                        "user_id": user["user_id"],
                        "session_id": session["session_id"],
                        "messages": session["messages"],
                    }
                    response = await client.post("/v1/memories/add", json=payload)
                    assert response.status_code == 200, response.text
            results = []
            for user in CORPUS["users"]:
                for question in user["questions"]:
                    request = {"user_id": user["user_id"], **question}
                    response = await client.post("/v1/memories/search", json=request)
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert list(body) == ["data"]
                    assert len(body["data"]) <= question["top_k"]
                    results.append({**request, "data": canonical(body["data"])})
        options = aml_options(memory).model_dump(mode="json")
    return {"search_options": options, "results": results}


def compare(name: str, actual: dict[str, Any]) -> None:
    path = SNAPSHOTS / f"{name}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    assert path.exists(), "no snapshot: run with MNEMOTREE_UPDATE_PARITY=1"
    assert actual == json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
async def test_engine_output_matches_parity_snapshot(tmp_path, name):
    actual = await run_engine(tmp_path, name)
    assert all(result["evidence"] for result in actual["results"]), "a question retrieves nothing"
    compare(name, actual)


async def test_service_output_matches_parity_snapshot(tmp_path):
    actual = await run_service(tmp_path)
    assert all(result["data"] for result in actual["results"]), "a question retrieves nothing"
    compare(SERVICE, actual)


async def test_corpus_exercises_the_preset_gates(tmp_path):
    """The histories must keep each history-dependent policy of the preset on both sides."""
    actual = await run_engine(tmp_path, "aml")
    preset = actual["search_options"]
    traces = {result["user_id"]: result["trace"] for result in actual["results"]}
    # Two speakers share the turns, so user turns are not preferred; in the
    # assistant-heavy histories they are.
    assert not traces["pair-chat"]["role_first_applied"]
    assert traces["pair-chat"]["assistant_share"] < preset["role_first_min_assistant_share"]
    assert traces["assistant-heavy"]["role_first_applied"]
    # Pasted reports several chunks long switch retrieval to the deeper pool.
    assert traces["assistant-heavy"]["candidate_limit"] == preset["candidate_limit"]
    assert traces["long-turns"]["chunk_density"] > preset["deep_history_density"]
    assert traces["long-turns"]["candidate_limit"] == preset["deep_candidate_limit"]


def test_every_snapshot_has_a_case():
    assert {path.stem for path in SNAPSHOTS.glob("*.json")} == {*CASES, SERVICE}
