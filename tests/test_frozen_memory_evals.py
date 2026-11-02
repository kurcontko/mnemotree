"""Prevent offline evals from passing on stale scores or incomplete fixtures."""

import copy
import importlib.util
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check_memory_evals.py"
SPEC = importlib.util.spec_from_file_location("check_memory_evals", SCRIPT)
evals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evals)


@pytest.fixture(scope="module")
def recording():
    return evals.load_fixture(evals.FIXTURES)


def case():
    return {
        "id": "locomo/example",
        "context_sha256": "old-context",
        "metrics": dict.fromkeys(evals.METRICS, 1.0),
    }


def test_changed_context_requires_fresh_scoring_but_is_not_a_retrieval_regression():
    old = case()
    current = {**old, "context_sha256": "new-context"}
    baseline = {"fixture_sha256": "fixture", "cases": [old]}
    regressions, changed = evals.compare([current], baseline, "fixture")
    assert not regressions
    assert changed == [old["id"]]


def test_lost_evidence_fails_even_when_the_context_hash_is_unchanged():
    old = case()
    current = copy.deepcopy(old)
    current["metrics"]["packed_turn_recall"] = 0.5
    baseline = {"fixture_sha256": "fixture", "cases": [old]}
    regressions, _ = evals.compare([current], baseline, "fixture")
    assert regressions == [
        {"id": old["id"], "metric": "packed_turn_recall", "baseline": 1.0, "actual": 0.5}
    ]


def test_missing_cases_cannot_pass():
    baseline = {"fixture_sha256": "fixture", "cases": [case()]}
    with pytest.raises(ValueError, match="question IDs differ"):
        evals.compare([], baseline, "fixture")


def test_lost_labels_cannot_turn_a_regression_into_an_unlabeled_case():
    old = case()
    current = copy.deepcopy(old)
    current["metrics"]["all_evidence"] = None
    baseline = {"fixture_sha256": "fixture", "cases": [old]}
    with pytest.raises(ValueError, match="label availability changed"):
        evals.compare([current], baseline, "fixture")


def test_baseline_for_another_model_fixture_is_rejected():
    baseline = {"fixture_sha256": "old-model", "cases": [case()]}
    with pytest.raises(ValueError, match="different fixture"):
        evals.compare([case()], baseline, "new-model")


async def test_exact_vectors_replay_and_changed_text_requires_refresh(recording):
    _, provenance, files = recording
    embedder = evals.FrozenEmbedder(provenance, files)
    question = evals.json.loads(files["questions.json"])[0]
    vector = await embedder.embed_query(question["text"])
    assert len(vector) == 256
    with pytest.raises(ValueError, match="Cache refresh required"):
        await embedder.embed_query(question["text"] + " changed query text")
    with pytest.raises(ValueError, match="Cache refresh required"):
        await embedder.embed_documents(["a new chunk that has no recorded vector"])


def test_corrupt_archive_fails_before_loading_vectors(tmp_path):
    (tmp_path / "fixture.json").write_bytes((evals.FIXTURES / "fixture.json").read_bytes())
    (tmp_path / "open-model-v1.zip").write_bytes(b"corrupt archive")
    with pytest.raises(ValueError, match="checksum changed"):
        evals.load_fixture(tmp_path)


def test_zero_vectors_cannot_disable_dense_retrieval(recording):
    _, provenance, files = recording
    invalid = {**files, "vectors.f32": bytes(len(files["vectors.f32"]))}
    with pytest.raises(ValueError, match="Invalid frozen vector"):
        evals.FrozenEmbedder(provenance, invalid)


def test_offline_guard_blocks_network_calls():
    with evals.offline(), pytest.raises(RuntimeError, match="forbids network access"):
        socket.create_connection(("example.invalid", 443))


def test_joined_context_and_block_token_counts_are_checked_separately():
    from mnemotree.evidence.models import Evidence, SearchOptions

    evidence = [
        Evidence(
            id=id,
            source_id=id,
            session_id="s",
            role="user",
            timestamp=None,
            start=0,
            end=len(text),
            text=text,
            score=1,
            channels={},
        )
        for id, text in (("a", "apple"), ("b", "banana"))
    ]
    sources = {
        item.source_id: {"content": item.text, "role": item.role, "session": item.session_id}
        for item in evidence
    }
    context = "\n\n".join(item.render() for item in evidence)
    trace = SimpleNamespace(
        mode="hybrid",
        channel_counts={"dense": 2},
        candidates=(),
        packed_tokens=len(context),
        evidence_tokens=sum(len(item.render()) for item in evidence),
    )
    recall = SimpleNamespace(context=context, evidence=evidence, items=(), trace=trace)
    labels = {"abstention": False, "evidence": [], "evidence_sessions": []}
    counter = SimpleNamespace(count=len)
    result = evals.inspect_recall(recall, sources, labels, SearchOptions(), counter)
    assert result["packed_tokens"] > result["evidence_tokens"]
    trace.packed_tokens -= 1
    with pytest.raises(ValueError, match="Packed token accounting differs"):
        evals.inspect_recall(recall, sources, labels, SearchOptions(), counter)
    trace.packed_tokens += 1
    trace.evidence_tokens -= 1
    with pytest.raises(ValueError, match="Evidence token accounting differs"):
        evals.inspect_recall(recall, sources, labels, SearchOptions(), counter)
