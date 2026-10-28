"""Smoke diagnostics must describe actual model inputs and returned evidence."""

import json
import subprocess

import httpx
import pytest

from mnemotree.evidence import Memory, Message, SearchOptions, cli
from mnemotree.evidence.models import ContextSpan, EmbeddingSpec, Evidence
from mnemotree.evidence.server import create_app, returned_role_chars


def test_role_char_counts_include_merged_neighbors_and_union_overlaps():
    def evidence(id, role, start, end):
        return Evidence(
            id=id,
            source_id=role,
            session_id="s",
            role=role,
            timestamp=None,
            start=start,
            end=end,
            text="x" * (end - start),
            score=1,
            channels={},
        )

    user = evidence("u", "user", 0, 10)
    assistant = evidence("a", "assistant", 0, 20)
    neighbor = ContextSpan(
        id="n",
        source_id="assistant",
        request_id="r",
        session_id="s",
        position=1,
        seq=1,
        offset=1,
        role="assistant",
        timestamp=None,
        start=10,
        end=30,
        text="x" * 20,
    )
    block = user.model_copy(update={"merged": (assistant,), "neighbors": (neighbor,)})
    overlap = evidence("a2", "assistant", 25, 40)
    assert returned_role_chars((block, overlap, user)) == {"user": 10, "assistant": 40}


@pytest.mark.parametrize("long_unicode", [False, True])
async def test_adapter_preserves_option_boundaries_and_model_input_and_logs(tmp_path, long_unicode):
    class Embedder:
        spec = EmbeddingSpec(provider="test", model="test", revision="1", dimensions=2)

        async def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        async def embed_query(self, text):
            self.query = text
            return [1.0, 0.0]

    embedder = Embedder()
    path = tmp_path / "requests.jsonl"
    with Memory(tmp_path / "m.db", embedder=embedder) as memory:
        await memory.add("u", [Message(content="zirconium")], request_id="r", session_id="s")
        app = create_app(memory, api_key="test", request_log=path)
        stem = " ".join(f"prefix{i}" for i in range(100))
        if long_unicode:
            stem += " 😃" * 5900
        options = [" ".join(f"choice{i}" for i in range(100)), "zirconium"]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer test"},
        ) as client:
            response = await client.post(
                "/v1/memories/search",
                json={
                    "user_id": "u",
                    "query": stem,
                    "options": options,
                    "top_k": 100,
                },
            )
        assert response.status_code == 200
        assert "zirconium" in response.json()["data"][0]["content"]
        # Dense retrieval also finds this source: the lexical channel count proves
        # the later option survived bounded term selection through the HTTP adapter.
        row = json.loads(path.read_text())
        assert row["channel_counts"]["lexical"] == 1
        full_query = stem + "\n" + "\n".join(options)
        assert row["original_model_query_tokens"] == memory.counter.count(full_query)
        assert row["model_query_tokens"] == memory.counter.count(embedder.query) <= 6000
        assert row["model_query_truncated"] is long_unicode
        if long_unicode:
            assert embedder.query.startswith("prefix0") and embedder.query.endswith("zirconium")
            assert row["model_query_tokens"] < row["original_model_query_tokens"]
        else:
            assert embedder.query == full_query
        assert row["assistant_share"] == 0
        assert row["role_first_applied"] is False
        assert row["returned_role_chars"] == {"user": len("zirconium")}
        assert row["tokenizer"] == memory.counter.signature
        assert "zirconium" not in path.read_text() and "prefix99" not in path.read_text()


async def test_role_gate_diagnostics_use_retrieval_snapshot(tmp_path):
    class WriteDuringRerank:
        written = False

        async def score(self, query, texts):
            if not self.written:
                self.written = True
                await memory.add(
                    "u",
                    [Message(role="assistant", content="answer " * 1000)],
                    request_id="later",
                    session_id="s",
                )
            return [1.0] * len(texts)

    with Memory(tmp_path / "m.db", reranker=WriteDuringRerank()) as memory:
        await memory.add("u", [Message(content="needle")], request_id="first", session_id="s")
        options = SearchOptions(
            mode="lexical", role_first="user", role_first_min_assistant_share=0.65
        )
        result = await memory.search("u", "needle", options=options)
        assert result.trace.assistant_share == 0
        assert result.trace.role_first_applied is False
        later = await memory.search("u", "needle", options=options)
        assert later.trace.assistant_share > 0.65
        assert later.trace.role_first_applied is True
        assert later.context == result.context


def test_code_identity_supports_installed_package_and_changed_sources(tmp_path, monkeypatch):
    package = tmp_path / "src" / "mnemotree" / "evidence"
    package.mkdir(parents=True)
    source = package / "cli.py"
    source.write_text("first\n")
    monkeypatch.setattr(cli, "__file__", str(source))
    monkeypatch.delenv("MNEMOTREE_BUILD_REVISION", raising=False)
    first = cli.code_identity()
    assert first["code_revision"] is None
    assert len(first["source_digest"]) == 64
    source.write_text("second\n")
    assert cli.code_identity()["source_digest"] != first["source_digest"]
    monkeypatch.setenv("MNEMOTREE_BUILD_REVISION", "a" * 40)
    assert cli.code_identity()["code_revision"] == "a" * 40


@pytest.mark.parametrize("git_missing", [False, True])
def test_code_identity_git_fallback_never_echoes_invalid_revision(
    tmp_path, monkeypatch, git_missing
):
    package = tmp_path / "src" / "mnemotree" / "evidence"
    package.mkdir(parents=True)
    source = package / "cli.py"
    source.write_text("source\n")
    (tmp_path / ".git").write_text("gitdir: /some/worktree\n")
    monkeypatch.setattr(cli, "__file__", str(source))
    monkeypatch.setenv("MNEMOTREE_BUILD_REVISION", "invalid-build-value")

    def run(args, **kwargs):
        assert args == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]
        if git_missing:
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(args, 0, stdout="b" * 40 + "\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.code_identity()["code_revision"] == (None if git_missing else "b" * 40)
