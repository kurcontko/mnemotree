"""Export a small offline fixture from the recorded local Qwen comparison.

Run with the recording's Python environment. Reads existing data and vectors;
never constructs a provider client or generates missing vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
import struct
import sys
import zipfile
from collections import defaultdict
from datetime import UTC
from pathlib import Path

SEED = "mnemotree-ci-v1"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DIMENSIONS = 256
TOKENIZER_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"
TOKENIZER_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def vector_key(kind: str, text: str) -> str:
    return sha(encoded([kind, text]))


def selection_key(value) -> str:
    return sha((SEED + ":" + value.id).encode())


def export(recording: Path, output: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "recorded_data", recording / "benchmarks/aml/data.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest = json.loads((recording / "manifest.json").read_bytes())
    for name in ("locomo", "lme"):
        source = Path(manifest["inventory"][name]["source"])
        with source.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != manifest["inventory"][name]["source_sha256"]:
            raise ValueError(f"Recorded {name} source hash changed")
    tokenizer = (Path.home() / ".cache/tiktoken" / TOKENIZER_CACHE_KEY).read_bytes()
    if sha(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("Recorded tokenizer hash changed")
    cache_path = recording / "vector-cache/shared.sqlite"
    with sqlite3.connect(cache_path.resolve().as_uri() + "?mode=ro", uri=True) as cache:
        identity = json.loads(
            cache.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
        )
        if identity["model"] != MODEL or identity["dimensions"] != 1024:
            raise ValueError("Expected the recorded native 1024-dimensional Qwen vectors")
        query_prefix = identity["query_prefix"]
        sources = {
            "locomo": module.load_locomo(manifest["inventory"]["locomo"]["source"]),
            "lme": module.load_longmemeval(manifest["inventory"]["lme"]["source"]),
        }
        histories, questions, labels, vectors = [], [], {}, {}

        def freeze(kind, text):
            key = vector_key(kind, text)
            if key in vectors:
                return
            provider_text = (query_prefix if kind == "query" else "") + text
            row = cache.execute(
                "SELECT vec FROM vectors WHERE key=?",
                (hashlib.sha256(provider_text.encode()).digest(),),
            ).fetchone()
            if row is None:
                raise ValueError(f"Cache refresh required: missing {kind} vector {key}")
            values = struct.unpack("<1024f", row[0])[:DIMENSIONS]
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Invalid recorded vector")
            vectors[key] = struct.pack(f"<{DIMENSIONS}f", *(value / norm for value in values))

        for dataset, samples in sources.items():
            chosen = []
            if dataset == "locomo":
                for sample in sorted(samples, key=selection_key)[:4]:
                    by_category = defaultdict(list)
                    for question in sample.questions:
                        by_category[question.category].append(question)
                    selected = [
                        question
                        for category in sorted(by_category)
                        for question in sorted(by_category[category], key=selection_key)[:2]
                    ]
                    chosen.append((sample, selected))
            else:
                by_category = defaultdict(list)
                for sample in samples:
                    by_category[sample.questions[0].category].append(sample)
                for category in sorted(by_category):
                    sample = min(by_category[category], key=selection_key)
                    chosen.append((sample, sample.questions))

            for sample, selected in chosen:
                history_id = f"{dataset}/{sample.id}"
                turns = []
                for turn in sample.turns:
                    # Preserve full haystacks, including distractors. The same
                    # projection was used by the original recorded experiment.
                    if turn.text.strip():
                        for start in range(0, len(turn.text), 1000):
                            freeze("document", turn.text[start : start + 1200])
                            if start + 1200 >= len(turn.text):
                                break
                    turns.append(
                        {
                            "id": turn.id,
                            "session": turn.session,
                            "role": turn.speaker
                            if turn.speaker in ("user", "assistant")
                            else "user",
                            "timestamp": turn.ts.replace(tzinfo=UTC).isoformat()
                            if turn.ts
                            else None,
                            "content": turn.text,
                        }
                    )
                histories.append({"id": history_id, "dataset": dataset, "messages": turns})
                known = {turn["id"] for turn in turns if turn["content"].strip()}
                for question in sorted(selected, key=lambda q: q.id):
                    qid = f"{dataset}/{question.id}"
                    freeze("query", question.question)
                    evidence = question.evidence - {""}
                    if not evidence <= known:
                        raise ValueError(f"Unresolved upstream evidence: {qid}: {evidence - known}")
                    questions.append(
                        {
                            "id": qid,
                            "history": history_id,
                            "text": question.question,
                            "visible": len(turns) if question.visible is None else question.visible,
                        }
                    )
                    labels[qid] = {
                        "category": question.category,
                        "evidence": sorted(evidence),
                        "evidence_sessions": sorted(question.evidence_sessions),
                        "abstention": question.category.endswith("_abs"),
                    }

    keys = sorted(vectors)
    files = {
        "histories.json": encoded(histories),
        "questions.json": encoded(questions),
        "labels.json": encoded(labels),
        "vector-keys.json": encoded(keys),
        "vectors.f32": b"".join(vectors[key] for key in keys),
        "tokenizer.tiktoken": tokenizer,
    }
    provenance = {
        "schema": 1,
        "selection": {
            "seed": SEED,
            "policy": "SHA256(seed + ':' + id); first 4 LoCoMo histories, first 2 questions per category per history; first LongMemEval history per category, including abstention",
            "uses_scores": False,
            "histories": len(histories),
            "questions": len(questions),
            "full_histories": True,
        },
        "recording": {
            "name": recording.name,
            "manifest_sha256": sha((recording / "manifest.json").read_bytes()),
            "loader_sha256": sha((recording / "benchmarks/aml/data.py").read_bytes()),
            "source_engine_revision": manifest["engine_revision"],
        },
        "datasets": {
            name: {
                "source_sha256": manifest["inventory"][name]["source_sha256"],
                "url": (
                    "https://github.com/snap-research/locomo/blob/main/data/locomo10.json"
                    if name == "locomo"
                    else "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
                ),
            }
            for name in sources
        },
        "embedding": {
            "provider": "frozen-open-model",
            "model": MODEL,
            "revision": REVISION,
            "dimensions": DIMENSIONS,
            "source_dimensions": 1024,
            "projection": "first 256 coordinates, L2 normalization, little-endian float32",
            "query_prefix": query_prefix,
            "document_prefix": "",
            "source_cache_identity": identity,
            "truncation": identity["embedding_truncation"],
            "license": "Apache-2.0",
            "vectors": len(keys),
        },
        "chunking": {"size": 1200, "overlap": 200},
        "tokenizer": {"encoding": "o200k_base", "cache_key": TOKENIZER_CACHE_KEY},
        "search": {
            "mode": "hybrid",
            "limit": 100,
            "candidate_limit": 100,
            "token_budget": 8000,
            "neighbor_messages": 1,
            "merge_blocks": True,
            "chunks_per_source": 1,
            "role_first": "user",
            "role_first_min_assistant_share": 0.2,
            "model_query_policy": "question",
        },
        "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in files.items()},
    }
    files["provenance.json"] = encoded(provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as handle,
        zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive,
    ):
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 2, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha(output.read_bytes()),
                "selection": provenance["selection"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.recording, args.output)
