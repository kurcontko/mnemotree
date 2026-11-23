"""The scripts in examples/ are the first code a new user copies, so they run here."""

import argparse
import importlib.util
from pathlib import Path

import pytest

from mnemotree import EvidenceEngine

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_multi_user_keeps_users_apart_and_forgets(tmp_path, capsys):
    await load("multi_user").main(tmp_path / "bot.sqlite")
    out = capsys.readouterr().out
    assert 'alice: ["I\'m allergic to peanuts."]' in out
    assert "bob: ['Peanuts are my favourite snack.']" in out
    assert "redelivery stored once: True" in out
    assert "conflict:" in out
    assert "after forget: 0 memories" in out


async def test_chat_recalls_what_an_earlier_run_was_told(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    chat = load("chat_with_memory")
    prompts = []

    async def complete(client, args, messages):
        prompts.append(messages)
        return "Noted."

    monkeypatch.setattr(chat, "complete", complete)
    args = argparse.Namespace(
        db=tmp_path / "chat.sqlite", user="me", config=None, model="m", base_url="http://test"
    )

    async def run(*lines):
        typed = iter(lines)

        def read(_prompt):
            try:
                return next(typed)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr("builtins.input", read)
        await chat.main(args)

    await run("I grow orchids on my balcony.", "")
    await run("What do I grow on my balcony?")

    first, second = prompts
    assert "orchids" not in first[0]["content"]
    assert "user: I grow orchids on my balcony." in second[0]["content"]
    # A new run starts a new conversation: only memory carries the earlier turn over.
    assert [message["role"] for message in second] == ["system", "user"]


async def test_service_backend_adds_and_searches_through_the_http_service(tmp_path, capsys):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from mnemotree.service.app import create_app

    with EvidenceEngine(tmp_path / "service.sqlite") as engine:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(engine, api_key="k")),
            base_url="http://test",
            headers={"Authorization": "Bearer k"},
        ) as client:
            await load("service_backend").main(client)
    out = capsys.readouterr().out
    assert out.startswith("stored: [")
    assert "user: My order arrived with a cracked screen." in out
