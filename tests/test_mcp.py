import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import mnemotree
from mnemotree.evidence import cli as evidence_cli
from mnemotree.mcp import TOOLS, Server, serve_stdio, tools
from mnemotree.mcp import server as mcp_server
from mnemotree.mcp.server import PROTOCOL_VERSIONS, main, open_engine


def request(id, method, params=None):
    message = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def call(id, name, arguments):
    return request(id, "tools/call", {"name": name, "arguments": arguments})


@pytest.fixture
def server(tmp_path):
    with mnemotree.Memory(tmp_path / "m.db", namespace="alice") as memory:
        yield Server(memory)


async def test_initialize_negotiates_the_protocol_version(server):
    reply = await server.handle(request(1, "initialize", {"protocolVersion": "2025-06-18"}))
    result = reply["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "mnemotree"
    reply = await server.handle(request(2, "initialize", {"protocolVersion": "1999-01-01"}))
    assert reply["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]
    assert await server.handle(request(3, "ping")) == {"jsonrpc": "2.0", "id": 3, "result": {}}


async def test_tools_list_advertises_self_contained_schemas(server):
    tools = (await server.handle(request(1, "tools/list")))["result"]["tools"]
    assert [tool["name"] for tool in tools] == list(TOOLS)
    for tool in tools:
        schema = json.dumps(tool["inputSchema"])
        assert "$ref" not in schema
        assert "$defs" not in schema
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"]["additionalProperties"] is False
    by_name = {tool["name"]: tool for tool in tools}
    assert by_name["recall"]["inputSchema"]["required"] == ["query"]
    assert by_name["recall"]["annotations"]["readOnlyHint"] is True
    assert by_name["forget"]["annotations"]["destructiveHint"] is True
    turn = by_name["remember_many"]["inputSchema"]["properties"]["messages"]["items"]
    assert turn["required"] == ["content"]


async def test_remember_recall_get_forget_through_tools(server):
    await server.handle(request(0, "initialize", {"protocolVersion": "2025-11-25"}))
    stored = await server.handle(
        call(
            1,
            "remember",
            {
                "content": "I grow orchids on my balcony.",
                "id": "orchids",
                "timestamp": "2026-10-28T09:30:00+01:00",
            },
        )
    )
    result = stored["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["memory"]["id"] == "orchids"
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    assert result["structuredContent"]["memory"]["timestamp"] == "2026-10-28T08:30:00Z"

    recalled = (await server.handle(call(2, "recall", {"query": "orchids"})))["result"]
    data = recalled["structuredContent"]
    assert [found["id"] for found in data["memories"]] == ["orchids"]
    assert "orchids on my balcony" in data["context"]
    assert data["packed_tokens"] > 0

    got = (await server.handle(call(3, "get", {"ids": ["orchids", "nope"]})))["result"]
    assert [found["id"] for found in got["structuredContent"]["memories"]] == ["orchids"]
    assert got["structuredContent"]["missing"] == ["nope"]

    gone = (await server.handle(call(4, "forget", {"id": "orchids"})))["result"]
    assert gone["structuredContent"] == {"forgotten": True}
    recalled = (await server.handle(call(5, "recall", {"query": "orchids"})))["result"]
    assert recalled["structuredContent"]["memories"] == []


async def test_remember_many_keeps_order(server):
    reply = await server.handle(
        call(
            1,
            "remember_many",
            {
                "session": "chat-1",
                "messages": [
                    {"content": "Which plant should I buy?", "id": "q"},
                    {"content": "An orchid.", "role": "assistant", "id": "a"},
                ],
            },
        )
    )
    memories = reply["result"]["structuredContent"]["memories"]
    assert [(m["id"], m["role"], m["session"]) for m in memories] == [
        ("q", "user", "chat-1"),
        ("a", "assistant", "chat-1"),
    ]


async def test_bad_arguments_and_conflicts_are_tool_errors(server):
    missing = (await server.handle(call(1, "recall", {})))["result"]
    assert missing["isError"] is True
    assert "query" in missing["content"][0]["text"]
    extra = (await server.handle(call(2, "recall", {"query": "x", "mode": "dense"})))["result"]
    assert extra["isError"] is True
    naive = (
        await server.handle(
            call(3, "remember", {"content": "x", "timestamp": "2026-10-28T09:30:00"})
        )
    )["result"]
    assert naive["isError"] is True
    assert "timezone" in naive["content"][0]["text"]
    await server.handle(call(4, "remember", {"content": "first", "request_id": "r"}))
    conflict = (await server.handle(call(5, "remember", {"content": "second", "request_id": "r"})))[
        "result"
    ]
    assert conflict["isError"] is True
    assert "different payload" in conflict["content"][0]["text"]


async def test_recall_limit_and_retried_writes_through_tools(server):
    await server.handle(request(0, "initialize", {"protocolVersion": "2025-11-25"}))
    write = {"content": "orchids and ferns", "request_id": "r"}
    first = (await server.handle(call(1, "remember", write)))["result"]["structuredContent"]
    retried = (await server.handle(call(2, "remember", write)))["result"]["structuredContent"]
    assert retried == first
    await server.handle(call(3, "remember", {"content": "more orchids"}))
    limited = (await server.handle(call(4, "recall", {"query": "orchids", "limit": 1})))["result"]
    assert len(limited["structuredContent"]["memories"]) == 1


async def test_oversized_arguments_are_tool_errors(server):
    too_long = "x" * (tools.MAX_CONTENT + 1)
    half = "x" * (tools.MAX_CONTENT // 2 + 1)
    calls = [
        call(1, "remember", {"content": too_long}),
        call(2, "recall", {"query": "x" * (tools.MAX_QUERY + 1)}),
        call(3, "remember_many", {"session": "s", "messages": [{"content": half}] * 2}),
        call(4, "remember_many", {"session": "s", "messages": [{"content": "x"}] * 1001}),
    ]
    for message in calls:
        assert (await server.handle(message))["result"]["isError"] is True
    recalled = (await server.handle(call(5, "recall", {"query": "x"})))["result"]
    assert recalled["structuredContent"]["memories"] == []


async def test_protocol_errors(server):
    unknown_tool = await server.handle(call(1, "nope", {}))
    assert unknown_tool["error"]["code"] == -32602
    unknown_method = await server.handle(request(2, "resources/list"))
    assert unknown_method["error"]["code"] == -32601
    assert (await server.handle({"jsonrpc": "1.0"}))["error"]["code"] == -32600
    notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert await server.handle(notification) is None
    assert await server.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None
    batch = await server.handle([request(3, "ping"), notification, request(4, "ping")])
    assert [reply["id"] for reply in batch] == [3, 4]


async def test_older_protocol_gets_text_only_results(server):
    await server.handle(request(1, "initialize", {"protocolVersion": "2025-03-26"}))
    result = (await server.handle(call(2, "remember", {"content": "x"})))["result"]
    assert "structuredContent" not in result
    assert json.loads(result["content"][0]["text"])["memory"]["content"] == "x"


async def test_stdio_loop_answers_each_line(server):
    stdin = io.BytesIO(
        b"\n".join(
            [
                json.dumps(request(1, "ping")).encode(),
                b"not json",
                b"",
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
                json.dumps(call(2, "remember", {"content": "naïve café"})).encode(),
            ]
        )
        + b"\n"
    )
    stdout = io.BytesIO()
    await serve_stdio(server, stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [reply.get("id") for reply in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700
    assert replies[2]["result"]["structuredContent"]["memory"]["content"] == "naïve café"


async def test_stdio_loop_survives_lines_it_cannot_answer_normally(server):
    too_deep = b"[" * 200_000 + b"]" * 200_000
    surrogate_id = b'{"jsonrpc": "2.0", "id": "\\ud800", "method": "ping"}'
    surrogate_method = b'{"jsonrpc": "2.0", "id": 1, "method": "\\ud800"}'
    ping = json.dumps(request(2, "ping")).encode()
    stdin = io.BytesIO(b"\n".join([too_deep, surrogate_id, surrogate_method, ping]) + b"\n")
    stdout = io.BytesIO()
    await serve_stdio(server, stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [reply.get("id") for reply in replies] == [None, "\ud800", 1, 2]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["result"] == {}
    assert replies[2]["error"]["code"] == -32601


def run_server(
    db: Path | str, *messages: dict, namespace: str = "t", extra: tuple[str, ...] = ()
) -> list[dict]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("MNEMOTREE_MCP")}
    env["PYTHONPATH"] = str(Path(mnemotree.__file__).parents[1])
    proc = subprocess.run(
        [sys.executable, "-m", "mnemotree.mcp", "--db", str(db), "--namespace", namespace]
        + ["--lexical", *extra],
        input="".join(json.dumps(message) + "\n" for message in messages),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    assert "Warning" not in proc.stderr, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_real_stdio_process_persists_across_restarts(tmp_path):
    db = tmp_path / "nested" / "m.db"
    initialize = request(0, "initialize", {"protocolVersion": "2025-11-25"})
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    first = run_server(
        db,
        initialize,
        initialized,
        request(1, "tools/list"),
        call(2, "remember", {"content": "The spare key is under the blue pot.", "id": "key"}),
    )
    assert [reply["id"] for reply in first] == [0, 1, 2]
    assert len(first[1]["result"]["tools"]) == len(TOOLS)
    second = run_server(db, initialize, initialized, call(1, "recall", {"query": "spare key"}))
    memories = second[1]["result"]["structuredContent"]["memories"]
    assert [found["id"] for found in memories] == ["key"]


def test_namespace_flag_scopes_the_process(tmp_path):
    db = tmp_path / "m.db"
    run_server(db, call(1, "remember", {"content": "Alice's note", "id": "n"}), namespace="alice")
    read = call(1, "get", {"ids": ["n"]})
    other = run_server(db, read, namespace="bob")[0]["result"]["structuredContent"]
    assert other["missing"] == ["n"]
    same = run_server(db, read, namespace="alice")[0]["result"]["structuredContent"]
    assert [memory["id"] for memory in same["memories"]] == ["n"]


def test_cli_expands_home_in_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "c.json").write_text(json.dumps({"search": {"mode": "lexical"}}))
    replies = run_server("~/nested/m.db", request(1, "ping"), extra=("--config", "~/c.json"))
    assert replies[0]["result"] == {}
    assert (tmp_path / "nested" / "m.db").exists()


def test_cli_requires_scope_and_an_explicit_retrieval_choice(tmp_path, capsys, monkeypatch):
    for name in ("MNEMOTREE_MCP_DB", "MNEMOTREE_MCP_NAMESPACE", "MNEMOTREE_MCP_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "m.db")])
    assert "--namespace" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "m.db"), "--namespace", "t"])
    assert "--lexical" in capsys.readouterr().err
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"embedding": {"kind": "sentence-transformers", "model": "x"}}))
    with pytest.raises(SystemExit):
        main(
            ["--db", str(tmp_path / "m.db"), "--namespace", "t", "--config", str(config)]
            + ["--lexical"]
        )
    assert "conflicts" in capsys.readouterr().err
    config.write_text(json.dumps({"search": {"limit": 5}}))
    with pytest.raises(SystemExit):
        main(
            ["--db", str(tmp_path / "m.db"), "--namespace", "t", "--config", str(config)]
            + ["--lexical"]
        )
    assert "search mode 'hybrid'" in capsys.readouterr().err
    assert not (tmp_path / "m.db").exists()


@pytest.mark.parametrize(
    "contents",
    ["{not json", '{"search": {"limit": 0}}', '{"embedding": {"kind": "nope"}}', None],
)
def test_cli_reports_a_bad_config_without_a_traceback(tmp_path, capsys, contents):
    config = tmp_path / "c.json"
    if contents is not None:
        config.write_text(contents)
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "m.db"), "--namespace", "t", "--config", str(config)])
    assert "--config:" in capsys.readouterr().err
    assert not (tmp_path / "m.db").exists()


def test_cli_refuses_a_bad_namespace_before_creating_the_database(tmp_path, capsys):
    db = tmp_path / "nested" / "m.db"
    with pytest.raises(SystemExit):
        main(["--db", str(db), "--namespace", "   ", "--lexical"])
    assert "nonblank" in capsys.readouterr().err
    assert not db.parent.exists()


def flaky_open(monkeypatch, errors):
    """Patch open_memory to raise ``errors`` in turn, then succeed; return the call log."""
    calls = []

    def fake(path, config):
        calls.append(path)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return "engine"

    monkeypatch.setattr(evidence_cli, "open_memory", fake)
    monkeypatch.setattr(mcp_server.time, "sleep", lambda seconds: None)
    return calls


def test_open_engine_retries_only_a_locked_database(tmp_path, monkeypatch):
    path, config = tmp_path / "m.db", evidence_cli.Config()
    locked = sqlite3.OperationalError("database is locked")
    calls = flaky_open(monkeypatch, [locked, locked])
    assert open_engine(path, config) == "engine"
    assert len(calls) == 3

    calls = flaky_open(monkeypatch, [sqlite3.OperationalError("disk I/O error")])
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        open_engine(path, config)
    assert len(calls) == 1


def test_open_engine_gives_up_after_its_attempts(tmp_path, monkeypatch):
    path, config = tmp_path / "m.db", evidence_cli.Config()
    calls = flaky_open(monkeypatch, [sqlite3.OperationalError("database is locked")] * 6)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        open_engine(path, config)
    assert len(calls) == 6


def test_separate_stdio_sessions_share_one_database(tmp_path):
    # Each Claude Code session spawns its own server; WAL lets them share the file.
    db = tmp_path / "shared.db"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MNEMOTREE_MCP")}
    env["PYTHONPATH"] = str(Path(mnemotree.__file__).parents[1])
    command = [sys.executable, "-m", "mnemotree.mcp", "--db", str(db), "--namespace", "t"]
    initialize = json.dumps(request(0, "initialize", {"protocolVersion": "2025-11-25"}))
    sessions = []
    for name in ("a", "b"):
        lines = [initialize] + [
            json.dumps(
                call(i, "remember", {"content": f"session {name} note {i}", "id": f"{name}{i}"})
            )
            for i in range(1, 21)
        ]
        proc = subprocess.Popen(
            [*command, "--lexical"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        sessions.append((proc, "\n".join(lines) + "\n"))
    for proc, payload in sessions:
        proc.stdin.write(payload)
        proc.stdin.flush()
    for proc, _ in sessions:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        replies = [json.loads(line) for line in out.splitlines()]
        assert all(not reply["result"].get("isError") for reply in replies[1:]), replies
    reader = run_server(db, request(0, "initialize", {}), call(1, "get", {"ids": ["a20", "b20"]}))
    found = reader[1]["result"]["structuredContent"]["memories"]
    assert sorted(memory["id"] for memory in found) == ["a20", "b20"]
