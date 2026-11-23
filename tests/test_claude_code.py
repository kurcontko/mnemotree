"""The Claude Code hooks: what they store, what they add to a prompt, and that they
never fail the session they run in."""

import io
import json
import os
import shutil
import sqlite3
import sys
import time
import types
from contextlib import closing

import pytest

from mnemotree import Memory
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


@pytest.fixture
def run(env, monkeypatch, capsys):
    """Send one hook event to main(). The detached flush runs in-process instead."""

    def flush_now(command, state):
        assert command[-2] == "--flush"
        hooks.main(list(command[command.index("hook") + 1 :]))

    monkeypatch.setattr(hooks, "detach", flush_now)

    def send(event, **fields):
        payload = {"hook_event_name": event, "session_id": "s1", **fields}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        hooks.main([], command=("mnemotree-mcp", "hook"))
        out = capsys.readouterr().out
        return json.loads(out) if out else None

    return send


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
    assert hooks.queued(env.state / "sessions" / "s1") == []


def test_a_later_session_gets_the_passage_and_the_same_one_only_once(run):
    turn(run, "s1", "We deploy to the VM harbor-7.", "Noted: harbor-7.")
    output = run("UserPromptSubmit", session_id="s2", prompt="Which VM do we deploy to?")
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    text = context(output)
    assert text.startswith("Mnemotree memory.")
    assert "user: We deploy to the VM harbor-7." in text
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
    assert hooks.queued(env.state / "sessions" / "s2") == []


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


def test_a_flush_that_died_after_its_write_does_not_store_twice(run, env, monkeypatch):
    session = env.state / "sessions" / "s1"
    run("UserPromptSubmit", prompt="We deploy to the VM harbor-7.")
    hooks.enqueue(session, "s1", "assistant", "Noted.")
    shutil.copytree(session / "queue", env.tmp / "saved")
    run("Stop")
    assert len(stored(env.db)) == 2
    # The process died before it removed the queue: the same files are flushed again.
    shutil.copytree(env.tmp / "saved", session / "queue", dirs_exist_ok=True)
    run("Stop")
    assert len(stored(env.db)) == 2


def test_stop_starts_a_detached_flush_with_the_hook_flags(env, monkeypatch, capsys):
    started = []
    monkeypatch.setattr(hooks, "detach", lambda command, state: started.append((command, state)))
    event = {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": "Done."}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    hooks.main(["--project"], command=("mnemotree-mcp", "hook"))
    assert started == [(["mnemotree-mcp", "hook", "--project", "--flush", "s1"], env.state)]
    assert capsys.readouterr().out == ""


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


def test_a_failed_store_keeps_the_queue_for_the_next_flush(run, env, monkeypatch):
    run("UserPromptSubmit", prompt="We deploy to the VM harbor-7.")
    (env.tmp / "file").write_text("not a directory")
    env.set("MNEMOTREE_DB", str(env.tmp / "file" / "m.db"))
    run("Stop", last_assistant_message="Noted.")
    session = env.state / "sessions" / "s1"
    assert len(hooks.queued(session)) == 2
    env.set("MNEMOTREE_DB", str(env.db))
    run("Stop")
    assert len(stored(env.db)) == 2
    assert hooks.queued(session) == []


def test_unknown_events_and_a_strange_session_id_are_handled(run, env):
    assert run("PreToolUse") is None
    run("UserPromptSubmit", session_id="../../escape", prompt="We deploy to the VM harbor-7.")
    (session,) = (env.state / "sessions").iterdir()
    assert len(session.name) == 64


def test_recall_waits_for_a_local_model_that_is_not_downloaded():
    def memory(embedder):
        return types.SimpleNamespace(engine=types.SimpleNamespace(embedder=embedder))

    assert hooks.ready(memory(None))
    assert hooks.ready(memory(types.SimpleNamespace(cached=lambda: True)))
    assert not hooks.ready(memory(types.SimpleNamespace(cached=lambda: False)))


def test_prune_removes_only_stale_sessions_with_an_empty_queue(env):
    sessions = env.state / "sessions"
    for name in ("old", "old-queued", "recent"):
        (sessions / name).mkdir(parents=True)
    hooks.enqueue(sessions / "old-queued", "old-queued", "user", "still waiting")
    month_ago = time.time() - hooks.STALE_SECONDS - 60
    for name in ("old", "old-queued"):
        os.utime(sessions / name, (month_ago, month_ago))
    hooks.prune(env.state)
    assert sorted(path.name for path in sessions.iterdir()) == ["old-queued", "recent"]


def test_the_detached_flush_stores_through_a_real_process(env, monkeypatch):
    run_event = {"hook_event_name": "UserPromptSubmit", "session_id": "s1"}
    for event in (
        {**run_event, "prompt": "We deploy to the VM harbor-7."},
        {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": "Noted."},
    ):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
        hooks.main([])
    deadline = time.monotonic() + 30
    while len(stored(env.db)) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert [row[1] for row in stored(env.db)] == ["user", "assistant"]


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
