"""Tests for LangChain tools with stubbed dependencies."""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


def _install_langchain_stubs() -> None:
    if importlib.util.find_spec("langchain") is not None:
        return

    langchain = types.ModuleType("langchain")
    langchain.__path__ = []

    tools = types.ModuleType("langchain.tools")

    @dataclass
    class Tool:
        func: callable
        coroutine: callable
        name: str
        description: str
        args_schema: type

        @classmethod
        def from_function(cls, *, func, coroutine, name, description, args_schema):
            return cls(
                func=func,
                coroutine=coroutine,
                name=name,
                description=description,
                args_schema=args_schema,
            )

    tools.Tool = Tool

    schema = types.ModuleType("langchain.schema")

    @dataclass
    class Document:
        page_content: str
        metadata: dict

    schema.Document = Document

    langchain.tools = tools
    langchain.schema = schema

    sys.modules["langchain"] = langchain
    sys.modules["langchain.globals"] = types.ModuleType("langchain.globals")
    sys.modules["langchain.tools"] = tools
    sys.modules["langchain.schema"] = schema


_install_langchain_stubs()

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.tools.langchain import LangchainMemoryTool, SearchMemoriesInput, StoreMemoryInput


class DummyMemoryCore:
    def __init__(self, recall_results=None, remember_result=None) -> None:
        self.recall_results = recall_results or []
        self.recall_calls: list[str] = []
        self.remember_calls: list[str] = []
        self.remember_result = remember_result

    async def recall(self, query: str):
        self.recall_calls.append(query)
        return list(self.recall_results)

    async def remember(self, data: str):
        self.remember_calls.append(data)
        return self.remember_result


@pytest.mark.asyncio
async def test_langchain_tool_search_memories():
    memory_core = DummyMemoryCore(
        [
            MemoryItem(
                content="hello",
                memory_type=MemoryType.SEMANTIC,
                importance=0.5,
            )
        ]
    )

    tool = LangchainMemoryTool(memory_core)
    tool.memory_formatter = MagicMock()
    tool.memory_formatter.format_memories.return_value = "formatted"

    result = await tool.asearch_memories("query")

    assert result == "formatted"
    assert memory_core.recall_calls == ["query"]
    tool.memory_formatter.format_memories.assert_called_once()


def test_langchain_tool_search_memories_sync(monkeypatch):
    memory_core = DummyMemoryCore(
        [
            MemoryItem(
                content="hello",
                memory_type=MemoryType.SEMANTIC,
                importance=0.5,
            )
        ]
    )

    tool = LangchainMemoryTool(memory_core)
    tool.memory_formatter = MagicMock()
    tool.memory_formatter.format_memories.return_value = "formatted"

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "run", lambda coro: loop.run_until_complete(coro))
    try:
        result = tool.search_memories("query")
    finally:
        loop.close()

    assert result == "formatted"
    assert memory_core.recall_calls == ["query"]


@pytest.mark.asyncio
async def test_langchain_tool_store_memory():
    stored = MagicMock()
    stored.memory_id = "mem-1"
    memory_core = DummyMemoryCore(remember_result=stored)

    tool = LangchainMemoryTool(memory_core)

    result = await tool.astore_memory("payload")

    assert result == "mem-1"
    assert memory_core.remember_calls == ["payload"]


def test_langchain_tool_store_memory_sync(monkeypatch):
    stored = MagicMock()
    stored.memory_id = "mem-2"
    memory_core = DummyMemoryCore(remember_result=stored)

    tool = LangchainMemoryTool(memory_core)
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "run", lambda coro: loop.run_until_complete(coro))
    try:
        result = tool.store_memory("payload")
    finally:
        loop.close()

    assert result == "mem-2"
    assert memory_core.remember_calls == ["payload"]


def test_langchain_tool_definitions():
    memory_core = DummyMemoryCore()
    tool = LangchainMemoryTool(memory_core)

    tools = tool.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "search_memories"
    assert tools[0].args_schema is SearchMemoriesInput
    assert tools[1].name == "store_memory"
    assert tools[1].args_schema is StoreMemoryInput
