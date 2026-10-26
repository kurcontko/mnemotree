"""Acceptance drills against real WAL, FTS5 and sqlite-vec files; no providers."""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from mnemotree import Memory, Message, SearchOptions
from mnemotree.evidence import EmbeddingSpec, IngestConflict


class DurabilityEmbedder:
    spec = EmbeddingSpec(provider="test", model="durability", revision="1", dimensions=2)

    async def embed_documents(self, texts):
        return [[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]

    async def embed_query(self, text):
        return (await self.embed_documents([text]))[0]


async def _seed(memory):
    await memory.add(
        "alice",
        [Message(id="shared", content="apple orchard café 華沙")],
        request_id="keep",
        session_id="session",
    )
    await memory.add(
        "alice",
        [Message(id="forgotten", content="erasedneedle pear")],
        request_id="deleted",
        session_id="session",
    )
    await memory.add(
        "bob",
        [Message(id="shared", content="apple private-bob")],
        request_id="keep",
        session_id="session",
    )
    assert memory.forget("alice", "forgotten")


async def _fingerprint(memory):
    result = {}
    for namespace in ("alice", "bob"):
        for mode in ("lexical", "dense", "hybrid"):
            found = await memory.search(namespace, "apple", options=SearchOptions(mode=mode))
            assert [item.source_id for item in found.evidence] == ["shared"]
            assert all(mode in item.channels or mode == "hybrid" for item in found.evidence)
            result[f"{namespace}/{mode}"] = {
                "context": found.context,
                "evidence": [item.model_dump(mode="json") for item in found.evidence],
                "tokens": found.trace.packed_tokens,
            }
    assert "private-bob" not in result["alice/hybrid"]["context"]
    assert "private-bob" in result["bob/hybrid"]["context"]
    assert not (
        await memory.search("alice", "erasedneedle", options=SearchOptions(mode="lexical"))
    ).evidence
    assert memory.source("alice", "forgotten") is None
    assert memory.source("unknown", "shared") is None
    return result


async def _check_receipts(memory):
    before = memory.store.stats("alice")
    replay = await memory.add(
        "alice",
        [Message(id="shared", content="apple orchard café 華沙")],
        request_id="keep",
        session_id="session",
    )
    assert replay.replayed
    with pytest.raises(IngestConflict, match="forgotten"):
        await memory.add(
            "alice",
            [Message(id="forgotten", content="erasedneedle pear")],
            request_id="deleted",
            session_id="session",
        )
    assert not memory.forget("alice", "forgotten")
    assert memory.store.stats("alice") == before


async def test_online_wal_backup_restores_indexes_scope_and_deletion_receipts(tmp_path):
    live, snapshot = tmp_path / "live.db", tmp_path / "restored.db"
    with Memory(live, embedder=DurabilityEmbedder()) as writer:
        writer.store.db.execute("PRAGMA wal_autocheckpoint=0")
        writer.store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await _seed(writer)
        expected = await _fingerprint(writer)
        assert writer.store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert Path(str(live) + "-wal").stat().st_size > 0
        # An online backup, rather than copying only the main file, includes
        # committed source, receipt, FTS and vector pages still held in the WAL.
        with closing(sqlite3.connect(snapshot)) as destination:
            writer.store.db.backup(destination)
        await writer.add(
            "alice",
            [Message(id="after-snapshot", content="apple later write")],
            request_id="later",
            session_id="session",
        )
        # Restore while the original writer remains open, at an independent path.
        with Memory(snapshot, embedder=DurabilityEmbedder()) as restored:
            assert await _fingerprint(restored) == expected
            assert restored.source("alice", "after-snapshot") is None
            assert writer.source("alice", "after-snapshot") is not None
            assert restored.store.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            await _check_receipts(restored)

    incompatible = DurabilityEmbedder()
    incompatible.spec = incompatible.spec.model_copy(update={"revision": "other-weights"})
    with pytest.raises(ValueError, match="embedding signature mismatch"):
        Memory(snapshot, embedder=incompatible)


def _child(script, database):
    return subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(database)],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def test_committed_wal_survives_unclean_exit_and_fresh_interpreter(tmp_path):
    database = tmp_path / "restart.db"
    # os._exit bypasses Python finalizers and SQLite close/checkpoint. This tests
    # an acknowledged WAL commit across process loss, not merely a reopen.
    seed = """
import asyncio, json, os, runpy, sys
ns = runpy.run_path(sys.argv[1])
memory = ns["Memory"](sys.argv[2], embedder=ns["DurabilityEmbedder"]())
memory.store.db.execute("PRAGMA wal_autocheckpoint=0")
asyncio.run(ns["_seed"](memory))
print(json.dumps(asyncio.run(ns["_fingerprint"](memory))), flush=True)
os._exit(0)
"""
    expected = json.loads(_child(seed, database))
    assert Path(str(database) + "-wal").stat().st_size > 0
    verify = """
import asyncio, json, runpy, sys
ns = runpy.run_path(sys.argv[1])
with ns["Memory"](sys.argv[2], embedder=ns["DurabilityEmbedder"]()) as memory:
    result = asyncio.run(ns["_fingerprint"](memory))
    asyncio.run(ns["_check_receipts"](memory))
    print(json.dumps(result))
"""
    assert json.loads(_child(verify, database)) == expected
