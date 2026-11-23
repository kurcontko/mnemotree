"""The Claude Code hooks: what they store, what they add to a prompt, and that they
never fail the session they run in."""

import asyncio
import io
import json
import os
import sqlite3
import sys
import time
import types
from contextlib import closing
from uuid import uuid4

import pytest

from mnemotree import Memory, Message
from mnemotree import claude_code as hooks
from mnemotree.mcp.project import project_namespace, project_root

pytest.importorskip("fcntl")


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A plugin data directory and database of the test's own, and no caller settings."""
    for name in list(os.environ):
        if name.startswith(("MNEMOTREE_", "CLAUDE_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "state"))
    monkeypatch.setenv("MNEMOTREE_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("MNEMOTREE_NAMESPACE", "project")
    monkeypatch.setenv("MNEMOTREE_LEXICAL", "1")
    return types.SimpleNamespace(
        state=tmp_path / "state", db=tmp_path / "m.db", set=monkeypatch.setenv, tmp=tmp_path
    )


def append_record(path, session, **fields):
    with path.open("a") as stream:
        stream.write(json.dumps({"sessionId": session, **fields}) + "\n")


@pytest.fixture
def client(env, monkeypatch, capsys):
    """Model the event IDs and transcript records emitted by Claude Code."""
    current = {}

    def send(
        event, *, accepted=True, answered=True, argv=(), command=("mnemotree-mcp", "hook"), **fields
    ):
        session = fields.get("session_id", "s1")
        path = env.tmp / (uuid4().hex + ".jsonl") if session not in current else current[session][1]
        if event == "UserPromptSubmit":
            current[session] = (fields.get("prompt_id", uuid4().hex), path)
        key, path = current.get(session, ("", path))
        payload = {
            "hook_event_name": event,
            "session_id": session,
            "prompt_id": key,
            "transcript_path": str(path),
            **fields,
        }
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        hooks.main(list(argv), command=command)
        if event == "UserPromptSubmit" and accepted:
            uid = uuid4().hex
            append_record(
                path,
                session,
                type="user",
                uuid=uid,
                promptId=key,
                message={"role": "user", "content": fields["prompt"]},
            )
            if answered:
                append_record(path, session, type="assistant", parentUuid=uid)
        out = capsys.readouterr().out
        return json.loads(out) if out else None

    return send


@pytest.fixture
def run(client, monkeypatch):
    def flush_now(command, state):
        assert command[-1] == "--flush"
        hooks.main(list(command[command.index("hook") + 1 :]))

    monkeypatch.setattr(hooks, "detach", flush_now)
    return client


def stored(db):
    if not db.exists():
        return []
    with closing(sqlite3.connect(db)) as connection:
        rows = connection.execute(
            "SELECT session_id, role, content FROM evidence_sources ORDER BY rowid"
        )
        return [tuple(row) for row in rows]


def turn(run, session, prompt, reply):
    run("UserPromptSubmit", session_id=session, prompt=prompt)
    run("Stop", session_id=session, last_assistant_message=reply)


def context(output):
    return output["hookSpecificOutput"]["additionalContext"]


def test_a_turn_is_stored_verbatim_as_two_messages_of_its_session(run, env):
    assert run("UserPromptSubmit", prompt="We deploy to the VM  harbor-7.") is None
    assert stored(env.db) == []  # the prompt waits for its reply
    assert run("Stop", last_assistant_message="Noted: harbor-7.") is None
    assert stored(env.db) == [
        ("s1", "user", "We deploy to the VM  harbor-7."),
        ("s1", "assistant", "Noted: harbor-7."),
    ]
    with hooks.HookState.open(env.state, env.db, "project") as state:
        assert not state.has_turns()


def test_a_later_session_gets_the_passage_and_the_same_one_only_once(run):
    turn(run, "s1", "We deploy to the VM harbor-7.", "Noted: harbor-7.")
    output = run("UserPromptSubmit", session_id="s2", prompt="Which VM do we deploy to?")
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    text = context(output)
    assert text.startswith("Mnemotree memory.")
    assert "user: We deploy to the VM harbor-7." in text
    run("Stop", session_id="s2", last_assistant_message="Thanks.")
    # Already in this session's context: asking again adds nothing.
    assert (
        run("UserPromptSubmit", session_id="s2", prompt="Which VM do we deploy to again?") is None
    )


def test_the_current_session_is_not_recalled_into_itself(run):
    turn(run, "s1", "We deploy to the VM harbor-7.", "Noted: harbor-7.")
    assert run("UserPromptSubmit", session_id="s1", prompt="Which VM do we deploy to?") is None


def test_slash_commands_and_blank_prompts_are_ignored(run, env):
    turn(run, "s1", "We deploy to the VM harbor-7.", "Noted.")
    for prompt in ("/clear the VM deploy notes", "   "):
        assert run("UserPromptSubmit", session_id="s2", prompt=prompt) is None
    run("Stop", session_id="s2", last_assistant_message="Command output.")
    assert all(row[0] != "s2" for row in stored(env.db))


def test_a_short_prompt_is_stored_but_not_searched(run, env):
    turn(run, "s1", "We deploy to the VM harbor-7.", "Noted.")
    assert run("UserPromptSubmit", session_id="s2", prompt="deploy VM") is None
    run("Stop", session_id="s2", last_assistant_message="To harbor-7.")
    assert ("s2", "user", "deploy VM") in stored(env.db)


def test_capture_and_recall_can_be_switched_off(run, env):
    turn(run, "s1", "We deploy to the VM harbor-7.", "Noted.")
    env.set("MNEMOTREE_RECALL_TOKENS", "0")
    assert run("UserPromptSubmit", session_id="s2", prompt="Which VM do we deploy to?") is None
    env.set("MNEMOTREE_RECALL_TOKENS", "500")
    env.set("MNEMOTREE_CAPTURE", "0")
    before = stored(env.db)
    assert run("UserPromptSubmit", session_id="s3", prompt="Which VM do we deploy to?")
    run("Stop", session_id="s3", last_assistant_message="harbor-7.")
    assert stored(env.db) == before


def test_recalled_text_stays_inside_the_token_budget(run, env):
    long_reply = "The VM deploy checklist: " + "check the harbor-7 disk and restart. " * 120
    turn(run, "s1", "Write the VM deploy checklist.", long_reply)
    turn(run, "s0", "We deploy to the VM harbor-7.", "Noted.")
    env.set("MNEMOTREE_RECALL_TOKENS", "60")
    text = context(run("UserPromptSubmit", session_id="s2", prompt="Which VM do we deploy to?"))
    body = text.removeprefix(hooks.HEADER).strip()
    assert "We deploy to the VM harbor-7." in body
    assert "restart" not in body
    with Memory(env.db, namespace="project") as memory:
        assert memory.engine.counter.count(body) <= 60


def test_an_interrupted_prompt_is_stored_with_the_next_reply(run, env):
    run("UserPromptSubmit", prompt="First question about the VM, interrupted.")
    turn(run, "s1", "Second question about the VM.", "The answer.")
    assert [row[1:] for row in stored(env.db)] == [
        ("user", "First question about the VM, interrupted."),
        ("user", "Second question about the VM."),
        ("assistant", "The answer."),
    ]


def test_a_flush_that_died_after_its_write_replays_with_a_new_turn(run, env, monkeypatch):
    original = Memory.remember_many

    async def fail_after_commit(self, *args, **kwargs):
        await original(self, *args, **kwargs)
        raise OSError("worker interrupted after commit")

    with monkeypatch.context() as patch:
        patch.setattr(Memory, "remember_many", fail_after_commit)
        turn(run, "s1", "First question", "First answer")
    assert len(stored(env.db)) == 2
    turn(run, "s1", "Second question", "Second answer")
    assert [row[2] for row in stored(env.db)] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
    ]
    with hooks.HookState.open(env.state, env.db, "project") as state:
        assert not state.has_turns()


def test_stop_starts_a_detached_flush_with_the_hook_flags(client, env, monkeypatch):
    started = []
    monkeypatch.setattr(hooks, "detach", lambda command, state: started.append((command, state)))
    client("UserPromptSubmit", prompt="A question", argv=["--project"])
    client("Stop", last_assistant_message="Done.", argv=["--project"])
    assert started == [(["mnemotree-mcp", "hook", "--project", "--flush"], env.state)]


def test_stop_without_anything_queued_starts_nothing(env, monkeypatch):
    monkeypatch.setattr(hooks, "detach", lambda *_: pytest.fail("started"))
    event = {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": "  "}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    hooks.main([])


@pytest.mark.parametrize(
    ("argv", "stdin", "logged"),
    [
        (["--no-such-flag"], "{}", "bad arguments"),
        ([], "not json", "JSONDecodeError"),
        (["--config", "/no/such/config.json"], "{}", "bad arguments"),
    ],
)
def test_a_failing_hook_exits_zero_and_logs(env, monkeypatch, capsys, argv, stdin, logged):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    # argparse would exit with 2, which blocks the user's prompt.
    hooks.main(argv)
    assert capsys.readouterr().out == ""
    assert logged in (env.state / "hook.log").read_text()


def test_another_session_retries_the_outbox_in_its_original_namespace(run, env, monkeypatch):
    async def offline(*args, **kwargs):
        raise OSError("offline")

    with monkeypatch.context() as patch:
        patch.setattr(Memory, "remember_many", offline)
        turn(run, "old", "Original note", "Original reply")
    with hooks.HookState.open(env.state, env.db, "project") as state:
        assert state.has_turns()
    env.set("MNEMOTREE_NAMESPACE", "other")
    turn(run, "new", "Other note", "Other reply")
    env.set("MNEMOTREE_NAMESPACE", "project")
    run("Stop", session_id="fresh")
    with closing(sqlite3.connect(env.db)) as db:
        assert db.execute(
            "SELECT namespace,content FROM evidence_sources ORDER BY rowid"
        ).fetchall() == [
            ("other", "Other note"),
            ("other", "Other reply"),
            ("project", "Original note"),
            ("project", "Original reply"),
        ]


def test_unconfirmed_prompts_never_enter_the_queue_or_database(run, env):
    run("UserPromptSubmit", prompt="An unaccepted note", accepted=False)
    # Some client versions retain a local record for a rejected prompt. No
    # assistant is linked to it, and the next Stop belongs to a different ID.
    run("UserPromptSubmit", prompt="Second unaccepted note", answered=False)
    for db_file in env.state.rglob("hooks.sqlite"):
        assert b"unaccepted note" not in db_file.read_bytes()
    turn(run, "s1", "  Accepted note\n", "    indented reply\n")
    assert [r[2] for r in stored(env.db)] == ["  Accepted note\n", "    indented reply\n"]


def test_missing_transcript_and_mismatched_stop_never_capture(run, env):
    run("UserPromptSubmit", prompt="Skipped question", accepted=False)
    run("Stop", prompt_id="unrelated", last_assistant_message="Wrong turn")
    run("Stop", last_assistant_message="No transcript")
    assert stored(env.db) == []


def test_an_older_client_needs_a_linked_assistant_to_confirm_the_user(run, env):
    run("UserPromptSubmit", prompt_id="", prompt="Unverified legacy turn", answered=False)
    run("Stop", last_assistant_message="Ignore this")
    assert stored(env.db) == []
    run("UserPromptSubmit", prompt_id="", prompt="Accepted legacy turn")
    run("Stop", last_assistant_message="Accepted answer")
    assert [r[2] for r in stored(env.db)] == ["Accepted legacy turn", "Accepted answer"]


def test_unknown_events_and_a_strange_session_id_are_handled(run, env):
    assert run("PreToolUse") is None
    turn(run, "../../escape", "This is a question.", "Done.")
    assert stored(env.db)[0] == ("../../escape", "user", "This is a question.")
    assert not (env.tmp / "escape").exists()


def test_recall_waits_for_a_local_model_that_is_not_downloaded():
    def memory(embedder):
        return types.SimpleNamespace(engine=types.SimpleNamespace(embedder=embedder))

    assert hooks.ready(memory(None))
    assert hooks.ready(memory(types.SimpleNamespace(cached=lambda: True)))
    assert not hooks.ready(memory(types.SimpleNamespace(cached=lambda: False)))


def test_prune_retains_confirmed_turns_waiting_for_storage(client, env, monkeypatch):
    monkeypatch.setattr(hooks, "detach", lambda *_: None)
    turn(client, "pending", "Confirmed prompt", "Confirmed reply")
    client("UserPromptSubmit", prompt="Still waiting")
    with hooks.HookState.open(env.state, env.db, "project") as state:
        with state.db:
            state.db.execute("UPDATE submissions SET submitted='2000-01-01'")
        state.prune()
        assert state.has_turns()
        assert state.db.execute("SELECT count(*) FROM submissions").fetchone()[0] == 0


def test_the_detached_flush_stores_through_a_real_process(client, env):
    client("UserPromptSubmit", prompt="We deploy to the VM harbor-7.")
    client(
        "Stop",
        last_assistant_message="Noted.",
        command=(sys.executable, "-m", "mnemotree.claude_code"),
    )
    deadline = time.monotonic() + 15
    while len(stored(env.db)) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert [row[1] for row in stored(env.db)] == ["user", "assistant"]


def test_filter_precedes_the_candidate_limit_and_keeps_unseen_chunks(run, env):
    config = env.tmp / "config.json"
    config.write_text(
        json.dumps(
            {
                "search": {
                    "mode": "lexical",
                    "limit": 1,
                    "candidate_limit": 1,
                    "deep_candidate_limit": 1,
                }
            }
        )
    )
    env.set("MNEMOTREE_CONFIG", str(config))
    with Memory(env.db, namespace="project") as memory:
        asyncio.run(
            memory.remember_many(
                [Message(content="Deploy checklist") for _ in range(15)], session="current"
            )
        )
        asyncio.run(
            memory.remember("Our deploy checklist includes Pacific approval.", session="older")
        )
        text = (
            "Quartz database alias is Pacific. "
            + "middle " * 220
            + "Zinnia observatory is Atlantic."
        )
        asyncio.run(memory.remember(text, session="older"))
    assert "Pacific approval" in context(
        run("UserPromptSubmit", session_id="current", prompt="deploy checklist status")
    )
    assert "Quartz" in context(
        run("UserPromptSubmit", session_id="current", prompt="quartz database alias")
    )
    run("Stop", session_id="current", last_assistant_message="Understood.")
    assert "Atlantic" in context(
        run("UserPromptSubmit", session_id="current", prompt="zinnia observatory Atlantic")
    )


def test_merged_blocks_and_their_neighbors_are_counted_as_delivered(run, env):
    cfg = env.tmp / "config.json"
    cfg.write_text(
        json.dumps({"search": {"mode": "lexical", "merge_blocks": True, "neighbor_messages": 1}})
    )
    env.set("MNEMOTREE_CONFIG", str(cfg))
    turn(run, "old", "The staging compiler is maple.", "The staging compiler target is silver.")
    text = context(run("UserPromptSubmit", session_id="new", prompt="staging compiler target"))
    assert "maple" in text and "silver" in text
    run("Stop", session_id="new", last_assistant_message="Thanks.")
    assert run("UserPromptSubmit", session_id="new", prompt="staging compiler target") is None


def test_rejected_attempt_does_not_hide_its_offered_passage(run):
    turn(run, "old", "The staging compiler is maple.", "Updated.")
    q = "which staging compiler"
    assert "maple" in context(run("UserPromptSubmit", session_id="new", prompt=q, accepted=False))
    assert "maple" in context(run("UserPromptSubmit", session_id="new", prompt=q))


def test_submit_uses_local_search_with_an_http_embedding_provider(run, env, monkeypatch):
    from mnemotree.providers.embeddings import OpenAIEmbedder

    embedded = []

    async def documents(self, texts):
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    async def query(self, text):
        pytest.fail("a submit hook contacted the embedding provider")

    monkeypatch.setattr(OpenAIEmbedder, "embed_documents", documents)
    monkeypatch.setattr(OpenAIEmbedder, "embed_query", query)
    env.set("MNEMOTREE_LEXICAL", "0")
    env.set("EMBEDDING_API_KEY", "test")
    config = env.tmp / "http.json"
    config.write_text(
        json.dumps(
            {"embedding": {"kind": "openai", "model": "fixture", "revision": "1", "dimensions": 2}}
        )
    )
    env.set("MNEMOTREE_CONFIG", str(config))
    turn(run, "prior", "Our staging compiler is maple.", "Recorded.")
    assert embedded == ["Our staging compiler is maple.", "Recorded."]
    assert "maple" in context(
        run("UserPromptSubmit", session_id="new", prompt="which staging compiler", accepted=False)
    )
    assert embedded == ["Our staging compiler is maple.", "Recorded."]


def test_forgetting_a_committed_turn_before_queue_ack_does_not_stall_new_turns(
    run, env, monkeypatch
):
    original = Memory.remember_many

    async def forget_after_commit(self, *args, **kwargs):
        items = await original(self, *args, **kwargs)
        await self.forget(items[0].id)
        raise OSError("interrupted before queue acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(Memory, "remember_many", forget_after_commit)
        turn(run, "s1", "Discarded original", "Earlier reply")
    turn(run, "s1", "Next question", "Next reply")
    assert [r[2] for r in stored(env.db)] == ["Earlier reply", "Next question", "Next reply"]


def test_transcript_matches_text_parts_and_ignores_unrelated_records(env):
    from mnemotree.claude_code.transcript import Submission, accepted_prompts, fingerprint

    prompt = "Keep the verbatim text."
    path = env.tmp / "transcript.jsonl"
    with path.open("w") as stream:
        stream.write("invalid JSON\n")
    record = {
        "type": "user",
        "uuid": "u1",
        "promptId": "p1",
        "message": {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
    }
    for extra in ({"sessionId": "other"}, {"isMeta": True}, {"isSidechain": True}):
        append_record(path, "s", **{**record, **extra})
    receipt = Submission("p1", fingerprint(prompt), 0, "2026-01-01T12:00:00+00:00")
    assert accepted_prompts(path, "s", [receipt]) is None
    append_record(path, "s", **record)
    uuid, messages = accepted_prompts(path, "s", [receipt])
    assert uuid == "u1" and [m.content for m in messages] == [prompt]


def test_project_root_is_the_repository_and_worktrees_share_it(tmp_path):
    plain = tmp_path / "plain" / "sub"
    plain.mkdir(parents=True)
    assert project_root(plain) == plain.resolve()

    repo = tmp_path / "repo"
    (repo / ".git" / "worktrees" / "feature").mkdir(parents=True)
    (repo / "src" / "pkg").mkdir(parents=True)
    assert project_root(repo / "src" / "pkg") == repo.resolve()

    worktree = tmp_path / "feature"
    (worktree / "docs").mkdir(parents=True)
    git_dir = repo / ".git" / "worktrees" / "feature"
    (git_dir / "commondir").write_text("../..\n")
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n")
    assert project_root(worktree / "docs") == repo.resolve()

    # A submodule's .git file has no commondir: it is its own project.
    module = repo / "vendor" / "lib"
    module.mkdir(parents=True)
    (repo / ".git" / "modules" / "lib").mkdir(parents=True)
    (module / ".git").write_text("gitdir: ../../.git/modules/lib\n")
    assert project_root(module) == module.resolve()


def test_project_namespace_is_stable_and_tells_same_named_checkouts_apart(tmp_path, monkeypatch):
    first, second = tmp_path / "a" / "api", tmp_path / "b" / "api"
    for path in (first, second):
        (path / ".git").mkdir(parents=True)
    assert project_namespace(first) == project_namespace(first / ".git")
    assert project_namespace(first).startswith("api-")
    assert project_namespace(first) != project_namespace(second)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(second))
    assert project_namespace() == project_namespace(second)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR")
    monkeypatch.chdir(first)
    assert project_namespace() == project_namespace(first)


def test_mnemotree_mcp_hook_uses_the_local_model_and_restarts_itself(env, monkeypatch):
    mnemotree_mcp = pytest.importorskip("mnemotree_mcp")
    calls = []
    monkeypatch.setattr(hooks, "main", lambda argv, **options: calls.append((argv, options)))
    assert mnemotree_mcp.main(["hook", "--project"]) == 0
    ((argv, options),) = calls
    assert argv == ["--project"]
    assert options["default_embedding"]["kind"] == "fastembed"
    assert options["command"] == (sys.executable, "-m", "mnemotree_mcp", "hook")
