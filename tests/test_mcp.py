import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import mnemotree
from mnemotree import Memory
from mnemotree.evidence import cli as evidence_cli
from mnemotree.mcp import TOOLS, Server, serve_stdio
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
    with Memory(tmp_path / "m.db", namespace="alice") as memory:
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
        assert "$ref" not in schema and "$defs" not in schema
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
    assert "orchids on my balcony" in data["context"] and data["packed_tokens"] > 0

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
    assert missing["isError"] is True and "query" in missing["content"][0]["text"]
    extra = (await server.handle(call(2, "recall", {"query": "x", "mode": "dense"})))["result"]
    assert extra["isError"] is True
    naive = (
        await server.handle(
            call(3, "remember", {"content": "x", "timestamp": "2026-10-28T09:30:00"})
        )
    )["result"]
    assert naive["isError"] is True and "timezone" in naive["content"][0]["text"]
    await server.handle(call(4, "remember", {"content": "first", "request_id": "r"}))
    conflict = (await server.handle(call(5, "remember", {"content": "second", "request_id": "r"})))[
        "result"
    ]
    assert conflict["isError"] is True and "different payload" in conflict["content"][0]["text"]


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


def run_server(db: Path, *messages: dict) -> list[dict]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("MNEMOTREE_MCP")}
    env["PYTHONPATH"] = str(Path(mnemotree.__file__).parents[1])
    proc = subprocess.run(
        [sys.executable, "-m", "mnemotree.mcp", "--db", str(db), "--namespace", "t"]
        + ["--lexical"],
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
    assert not (tmp_path / "m.db").exists()


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
    locked = sqlite3.OperationalError("database is locked")
    calls = flaky_open(monkeypatch, [locked, locked])
    assert open_engine(tmp_path / "m.db", evidence_cli.Config()) == "engine"
    assert len(calls) == 3

    calls = flaky_open(monkeypatch, [sqlite3.OperationalError("disk I/O error")])
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        open_engine(tmp_path / "m.db", evidence_cli.Config())
    assert len(calls) == 1


def test_open_engine_gives_up_after_its_attempts(tmp_path, monkeypatch):
    calls = flaky_open(monkeypatch, [sqlite3.OperationalError("database is locked")] * 6)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        open_engine(tmp_path / "m.db", evidence_cli.Config())
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
        proc.stdin.close()
    for proc, _ in sessions:
        with proc.stdout, proc.stderr:
            replies = [json.loads(line) for line in proc.stdout.read().splitlines()]
            assert proc.wait(timeout=60) == 0, proc.stderr.read()
        assert all(not reply["result"].get("isError") for reply in replies[1:]), replies
    reader = run_server(db, request(0, "initialize", {}), call(1, "get", {"ids": ["a20", "b20"]}))
    found = reader[1]["result"]["structuredContent"]["memories"]
    assert sorted(memory["id"] for memory in found) == ["a20", "b20"]
