"""Replay real LoCoMo/LongMemEval histories with frozen open model vectors.

The candidate is the current checkout. Only provider data and retrieval floors
are frozen; contexts are generated anew. No answer accuracy is inferred from a
historical reader response after the context changes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import struct
import tempfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from mnemotree import Memory
from mnemotree.evidence.models import EmbeddingSpec, Message, SearchOptions
from mnemotree.evidence.packing import TiktokenCounter

FIXTURES = Path(__file__).resolve().parents[2] / "benchmarks/ci/fixtures"
METRICS = (
    "candidate_turn_recall",
    "packed_turn_recall",
    "all_evidence",
    "packed_session_recall",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vector_key(kind: str, text: str) -> str:
    return sha(json.dumps([kind, text], ensure_ascii=False, separators=(",", ":")).encode())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@contextmanager
def offline():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Frozen evaluation forbids network access")

    with ExitStack() as stack:
        for owner, name in (
            (socket, "create_connection"),
            (socket, "getaddrinfo"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
        ):
            stack.enter_context(patch.object(owner, name, forbidden))
        yield


def load_fixture(directory: Path):
    pin = json.loads((directory / "fixture.json").read_bytes())
    archive_bytes = (directory / "open-model-v1.zip").read_bytes()
    require(sha(archive_bytes) == pin["sha256"], "Frozen fixture archive checksum changed")
    with zipfile.ZipFile(directory / "open-model-v1.zip") as archive:
        provenance = json.loads(archive.read("provenance.json"))
        require(provenance["schema"] == 1, "Unsupported frozen fixture schema")
        names = {"provenance.json", *provenance["files"]}
        require(set(archive.namelist()) == names, "Frozen fixture file inventory changed")
        files = {}
        for name, expected in provenance["files"].items():
            data = archive.read(name)
            require(
                len(data) == expected["bytes"] and sha(data) == expected["sha256"],
                f"Frozen fixture checksum changed: {name}",
            )
            files[name] = data
    return pin, provenance, files


class FrozenEmbedder:
    def __init__(self, provenance, files):
        config = provenance["embedding"]
        self.spec = EmbeddingSpec(
            **{key: config[key] for key in ("provider", "model", "revision", "dimensions")}
        )
        keys = json.loads(files["vector-keys.json"])
        require(len(keys) == len(set(keys)) == config["vectors"], "Invalid vector inventory")
        size = self.spec.dimensions * 4
        data = files["vectors.f32"]
        require(len(data) == len(keys) * size, "Invalid frozen vector byte length")
        self.vectors = {}
        for index, key in enumerate(keys):
            values = struct.unpack_from(f"<{self.spec.dimensions}f", data, index * size)
            norm = math.sqrt(sum(value * value for value in values))
            require(math.isfinite(norm) and abs(norm - 1) < 1e-5, "Invalid frozen vector")
            self.vectors[key] = values
        self.document_lookups = 0
        self.query_lookups = 0

    def lookup(self, kind, text):
        key = vector_key(kind, text)
        if key not in self.vectors:
            raise ValueError(f"Cache refresh required: missing {kind} vector {key}")
        return self.vectors[key]

    async def embed_documents(self, texts):
        self.document_lookups += len(texts)
        return [self.lookup("document", text) for text in texts]

    async def embed_query(self, text):
        self.query_lookups += 1
        return self.lookup("query", text)


def coverage(expected, actual):
    return len(set(expected) & actual) / len(expected) if expected else None


def inspect_recall(recall, sources, labels, options, counter):
    require(recall.trace.mode == "hybrid", "Hybrid retrieval silently changed mode")
    require(recall.trace.channel_counts.get("dense", 0) > 0, "Dense retrieval did not run")
    require(len(recall.evidence) <= options.limit, "Returned item limit exceeded")
    require(
        recall.context == "\n\n".join(item.render() for item in recall.evidence),
        "Packed context differs from rendered evidence",
    )
    tokens = counter.count(recall.context)
    evidence_tokens = sum(counter.count(item.render()) for item in recall.evidence)
    require(tokens == recall.trace.packed_tokens, "Packed token accounting differs")
    require(evidence_tokens == recall.trace.evidence_tokens, "Evidence token accounting differs")
    require(tokens <= options.token_budget, "Packed token budget exceeded")
    require(evidence_tokens <= options.token_budget, "Evidence token budget exceeded")
    packed = set()
    for item in recall.evidence:
        for member in item.members():
            for span in (member, *member.neighbors):
                require(span.source_id in sources, "Source outside visible history")
                source = sources[span.source_id]
                require(0 <= span.start < span.end <= len(source["content"]), "Invalid source span")
                require(
                    span.text == source["content"][span.start : span.end],
                    "Packed evidence differs from original source",
                )
                require(span.role == source["role"], "Packed evidence role differs")
                packed.add(span.source_id)
    for item in recall.items:
        require(item.id in sources, "Returned memory outside visible history")
        require(item.content == sources[item.id]["content"], "Returned memory differs from source")
    candidates = {candidate.chunk.source_id for candidate in recall.trace.candidates}
    require(candidates <= sources.keys(), "Candidate outside visible history")
    evidence = [] if labels["abstention"] else labels["evidence"]
    sessions = [] if labels["abstention"] else labels["evidence_sessions"]
    metrics = {
        "candidate_turn_recall": coverage(evidence, candidates),
        "packed_turn_recall": coverage(evidence, packed),
        "all_evidence": float(set(evidence) <= packed) if evidence else None,
        "packed_session_recall": coverage(sessions, {sources[id]["session"] for id in packed}),
    }
    return {
        "metrics": metrics,
        "context_sha256": sha(recall.context.encode()),
        "packed_tokens": tokens,
        "evidence_tokens": evidence_tokens,
        "packed_source_ids": sorted(packed),
        "candidate_source_ids": sorted(candidates),
    }


async def replay(provenance, files, working: Path):
    histories = json.loads(files["histories.json"])
    questions = json.loads(files["questions.json"])
    labels = json.loads(files["labels.json"])
    require(len(histories) == provenance["selection"]["histories"], "History inventory differs")
    require(len(questions) == provenance["selection"]["questions"], "Question inventory differs")
    ids = [question["id"] for question in questions]
    require(len(set(ids)) == len(ids) and set(ids) == labels.keys(), "Question/label IDs differ")
    history_ids = [history["id"] for history in histories]
    require(len(history_ids) == len(set(history_ids)), "Duplicate history IDs")
    require({q["history"] for q in questions} == set(history_ids), "Question histories differ")
    tokenizer_cache = working / "tokenizer"
    tokenizer_cache.mkdir()
    (tokenizer_cache / provenance["tokenizer"]["cache_key"]).write_bytes(
        files["tokenizer.tiktoken"]
    )
    with patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": str(tokenizer_cache)}):
        counter = TiktokenCounter(provenance["tokenizer"]["encoding"])
    embedder = FrozenEmbedder(provenance, files)
    options = SearchOptions.model_validate(provenance["search"])
    chunking = provenance["chunking"]
    rows = []
    # Share a database so real histories exercise namespace isolation as well.
    with Memory(
        working / "memory.sqlite",
        namespace="bootstrap",
        embedder=embedder,
        token_counter=counter,
        chunk_size=chunking["size"],
        chunk_overlap=chunking["overlap"],
    ) as memory:
        for history in histories:
            bound = Memory.from_engine(memory.engine, namespace=history["id"])
            history_questions = sorted(
                (q for q in questions if q["history"] == history["id"]),
                key=lambda q: (q["visible"], q["id"]),
            )
            ingested, sources = 0, {}
            for question in history_questions:
                visible = question["visible"]
                require(ingested <= visible <= len(history["messages"]), "Invalid visible prefix")
                by_session = defaultdict(list)
                for turn in history["messages"][ingested:visible]:
                    if turn["content"].strip():
                        require(turn["id"] not in sources, "Duplicate source ID within history")
                        sources[turn["id"]] = turn
                        by_session[turn["session"]].append(
                            Message(
                                id=turn["id"],
                                content=turn["content"],
                                role=turn["role"],
                                timestamp=datetime.fromisoformat(turn["timestamp"])
                                if turn["timestamp"]
                                else None,
                            )
                        )
                for session, messages in by_session.items():
                    for start in range(0, len(messages), 20):
                        await bound.remember_many(
                            messages[start : start + 20],
                            session=session,
                            request_id=f"ci:{ingested}:{session}:{start}",
                        )
                ingested = visible
                label = labels[question["id"]]
                require(set(label["evidence"]) <= sources.keys(), "Evidence outside visible prefix")
                result = await bound.recall(question["text"], options=options)
                row = inspect_recall(result, sources, label, options, counter)
                rows.append(
                    {
                        "id": question["id"],
                        "dataset": history["dataset"],
                        "category": label["category"],
                        **row,
                    }
                )
    return rows, {
        "document_vectors": embedder.document_lookups,
        "query_vectors": embedder.query_lookups,
    }


def compare(rows, baseline, fixture_sha):
    require(baseline["fixture_sha256"] == fixture_sha, "Baseline belongs to a different fixture")
    expected = {row["id"]: row for row in baseline["cases"]}
    require(len(expected) == len(baseline["cases"]), "Duplicate baseline IDs")
    require({row["id"] for row in rows} == expected.keys(), "Baseline question IDs differ")
    regressions, changed = [], []
    for row in rows:
        old = expected[row["id"]]
        if row["context_sha256"] != old["context_sha256"]:
            changed.append(row["id"])
        for metric in METRICS:
            actual, floor = row["metrics"][metric], old["metrics"][metric]
            require((actual is None) == (floor is None), "Metric label availability changed")
            if floor is not None and actual + 1e-12 < floor:
                regressions.append(
                    {"id": row["id"], "metric": metric, "baseline": floor, "actual": actual}
                )
    return regressions, changed


def summary(rows):
    lines = [
        "# Frozen memory evaluations",
        "",
        "| Dataset | Cases | Packed turn recall | All evidence |",
        "|---|---:|---:|---:|",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        cells = []
        for metric in ("packed_turn_recall", "all_evidence"):
            values = [
                row["metrics"][metric] for row in selected if row["metrics"][metric] is not None
            ]
            cells.append(
                f"{sum(values) / len(values):.1%} ({len(values)} labeled)"
                if values
                else "unlabeled"
            )
        lines.append(f"| {dataset} | {len(selected)} | {' | '.join(cells)} |")
    lines.extend(["", "These are source-turn coverage measurements, not answer accuracy."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-baseline",
        type=Path,
        help="Write a new candidate baseline for explicit review; never overwrites a file",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with offline(), tempfile.TemporaryDirectory(prefix="mnemotree-evals-") as temporary:
            pin, provenance, files = load_fixture(args.fixtures)
            rows, lookups = asyncio.run(replay(provenance, files, Path(temporary)))
        candidate = {"fixture_sha256": pin["sha256"], "cases": rows}
        if args.record_baseline:
            with args.record_baseline.open("x") as handle:
                json.dump(candidate, handle, indent=2)
                handle.write("\n")
            regressions, changed = [], []
        else:
            baseline = json.loads((args.fixtures / "baseline.json").read_bytes())
            regressions, changed = compare(rows, baseline, pin["sha256"])
        report = {
            **candidate,
            "valid": not regressions,
            "regressions": regressions,
            "changed_contexts": changed,
            "lookups": lookups,
            "model_calls": 0,
        }
        text = summary(rows)
        text += f"\nRegressions: {len(regressions)}. Changed contexts: {len(changed)}. Model calls: 0.\n"
        if regressions:
            text += (
                "\n"
                + "\n".join(
                    f"- {row['id']}: {row['metric']} {row['baseline']:.3f} → {row['actual']:.3f}"
                    for row in regressions
                )
                + "\n"
            )
    except (ValueError, RuntimeError, OSError, KeyError, zipfile.BadZipFile) as error:
        report = {"valid": False, "error": str(error)}
        text = f"Frozen memory evaluation failed: {error}\n"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.md").write_text(text)
    print(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
