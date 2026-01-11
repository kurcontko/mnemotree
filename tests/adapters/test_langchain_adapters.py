"""Tests for LangChain adapters with stubbed dependencies."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import types
import importlib.util
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import asyncio
from uuid import uuid4


def _install_langchain_core_stubs() -> None:
    if importlib.util.find_spec("langchain_core") is not None:
        return

    langchain_core = types.ModuleType("langchain_core")
    langchain_core.__path__ = []

    callbacks = types.ModuleType("langchain_core.callbacks")

    class CallbackManagerForRetrieverRun:
        pass

    callbacks.CallbackManagerForRetrieverRun = CallbackManagerForRetrieverRun

    documents = types.ModuleType("langchain_core.documents")

    @dataclass
    class Document:
        page_content: str
        metadata: dict

    documents.Document = Document

    retrievers = types.ModuleType("langchain_core.retrievers")

    class BaseRetriever:
        def __init__(self, *args, **kwargs) -> None:
            """Mock implementation for testing."""
            pass

    retrievers.BaseRetriever = BaseRetriever

    memory = types.ModuleType("langchain_core.memory")

    class BaseMemory:
        pass

    memory.BaseMemory = BaseMemory

    langchain_core.callbacks = callbacks
    langchain_core.documents = documents
    langchain_core.retrievers = retrievers
    langchain_core.memory = memory

    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.callbacks"] = callbacks
    sys.modules["langchain_core.documents"] = documents
    sys.modules["langchain_core.retrievers"] = retrievers
    sys.modules["langchain_core.memory"] = memory


_install_langchain_core_stubs()

from mnemotree.adapters.langchain.memory import MemoryLangChainAdapter
from mnemotree.adapters.langchain.retriever import MemoryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class DummyMemoryCore:
    def __init__(self, recall_results=None) -> None:
        self.recall_results = recall_results or []
        self.recall_calls: list[tuple[str, int]] = []
        self.remember_calls: list[str] = []

    async def recall(self, *, query: str, limit: int):
        self.recall_calls.append((query, limit))
        return list(self.recall_results)

    async def remember(self, content: str):
        self.remember_calls.append(content)


def _make_run_manager() -> CallbackManagerForRetrieverRun:
    try:
        return CallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
    except TypeError:
        return CallbackManagerForRetrieverRun()


@pytest.mark.asyncio
async def test_memory_retriever_async_recall():
    doc = Document(page_content="hello", metadata={})
    memory_item = MagicMock()
    memory_item.to_langchain_document.return_value = doc
    memory_core = DummyMemoryCore([memory_item])

    retriever = MemoryRetriever(memory_core)
    results = await retriever._aget_relevant_documents(
        "query",
        run_manager=_make_run_manager(),
        k=1,
    )

    assert results == [doc]
    assert memory_core.recall_calls == [("query", 1)]


def test_memory_retriever_sync_recall(monkeypatch):
    doc = Document(page_content="hello", metadata={})
    memory_item = MagicMock()
    memory_item.to_langchain_document.return_value = doc
    memory_core = DummyMemoryCore([memory_item])

    retriever = MemoryRetriever(memory_core)
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "run", lambda coro: loop.run_until_complete(coro))
    try:
        results = retriever._get_relevant_documents(
            "query",
            run_manager=_make_run_manager(),
            k=1,
        )
    finally:
        loop.close()

    assert results == [doc]
    assert memory_core.recall_calls == [("query", 1)]


@pytest.mark.asyncio
async def test_memory_adapter_loads_history():
    memory_one = MagicMock()
    memory_one.to_str_llm.return_value = "Memory A"
    memory_two = MagicMock()
    memory_two.to_str_llm.return_value = "Memory B"
    memory_core = DummyMemoryCore([memory_one, memory_two])

    adapter = MemoryLangChainAdapter(memory_core)
    result = await adapter.aload_memory_variables({"input": "hi"})

    assert result["history"] == "Memory A\n\nMemory B"
    assert memory_core.recall_calls == [("hi", 10)]


@pytest.mark.asyncio
async def test_memory_adapter_saves_context():
    memory_core = DummyMemoryCore()
    adapter = MemoryLangChainAdapter(memory_core)

    await adapter.asave_context({"input": "hello"}, {"output": "world"})

    assert memory_core.remember_calls == ["User: hello\nAssistant: world"]
