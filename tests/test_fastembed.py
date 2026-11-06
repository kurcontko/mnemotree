import asyncio
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

from mnemotree.config import Config, open_memory
from mnemotree.providers.embeddings import (
    DEFAULT_LOCAL_EMBEDDING,
    EmbedderUnavailable,
    FastEmbedEmbedder,
)

TOY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
]


def embedder(fake, cache_dir, **overrides):
    settings = fake.settings(cache_dir) | overrides
    settings.pop("kind")
    return FastEmbedEmbedder(**settings)


def test_spec_is_ready_without_loading_and_the_cache_expands_home(
    fake_fastembed, tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    model = embedder(fake_fastembed, "~/models", query_prompt="q: ", document_prompt="d: ")
    assert model.cache_dir == tmp_path / "models"
    assert model.spec.provider == "fastembed"
    assert model.spec.model == "toy/embedder"
    assert model.spec.dimensions == 4
    assert model.spec.revision == (
        "toy/embedder-onnx@0123456789abcdef0123456789abcdef01234567|query=q: |document=d: "
    )
    assert fake_fastembed.downloads == [] and fake_fastembed.loads == []


async def test_concurrent_first_calls_download_and_load_once(fake_fastembed, tmp_path):
    model = embedder(fake_fastembed, tmp_path / "models")
    fake_fastembed.release.clear()
    calls = [asyncio.create_task(model.embed_documents(["abc", "de"])) for _ in range(8)]
    calls.append(asyncio.create_task(model.embed_query("abcd")))
    await asyncio.sleep(0.05)
    fake_fastembed.release.set()
    results = await asyncio.gather(*calls)
    assert len(fake_fastembed.loads) == 1
    assert [call.get("local_files_only") for call in fake_fastembed.downloads] == [True, None]
    documents, query = results[0], results[-1]
    assert documents[0] == pytest.approx([3 / math.sqrt(18), 3 / math.sqrt(18), 0, 0])
    assert query == pytest.approx([0.8, 0.6, 0, 0])
    assert all(math.isclose(math.hypot(*vector), 1) for vector in documents)


async def test_fetches_the_pinned_files_once_then_reads_the_cache(fake_fastembed, tmp_path):
    cache = tmp_path / "models"
    await embedder(fake_fastembed, cache).embed_query("x")
    local, network = fake_fastembed.downloads
    expected = {
        "repo_id": "toy/embedder-onnx",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "cache_dir": str(cache),
        "allow_patterns": TOY_FILES,
    }
    assert local == expected | {"local_files_only": True}
    assert network == expected
    snapshot = str(cache / "snapshots" / expected["revision"])
    assert fake_fastembed.loads == [
        {"model": "toy/embedder", "cache_dir": str(cache), "path": snapshot}
    ]
    # A restart finds every file in the cache and never asks the network.
    fake_fastembed.downloads.clear()
    await embedder(fake_fastembed, cache).embed_query("x")
    assert [call.get("local_files_only") for call in fake_fastembed.downloads] == [True]


async def test_an_interrupted_download_is_fetched_again(fake_fastembed, tmp_path):
    cache = tmp_path / "models"
    await embedder(fake_fastembed, cache).embed_query("x")
    snapshot = cache / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
    (snapshot / "onnx" / "model.onnx_data").unlink()
    fake_fastembed.downloads.clear()
    await embedder(fake_fastembed, cache).embed_query("x")
    assert [call.get("local_files_only") for call in fake_fastembed.downloads] == [True, None]


async def test_dimension_mismatches_are_clear_errors(fake_fastembed, tmp_path):
    with pytest.raises(ValueError, match="has 4 dimensions, not the configured 8"):
        await embedder(fake_fastembed, tmp_path, dimensions=8).embed_query("x")
    assert fake_fastembed.downloads == []  # Caught before downloading anything.

    fake_fastembed.embedding_size = 8
    with pytest.raises(ValueError, match="loaded with 8 dimensions"):
        await embedder(fake_fastembed, tmp_path).embed_query("x")

    fake_fastembed.embedding_size, fake_fastembed.vector_size = 4, 6
    with pytest.raises(ValueError, match="returned 6 dimensions"):
        await embedder(fake_fastembed, tmp_path).embed_query("x")


async def test_unsupported_model_is_rejected(fake_fastembed, tmp_path):
    with pytest.raises(ValueError, match="does not support"):
        await embedder(fake_fastembed, tmp_path, model="toy/other").embed_query("x")


async def test_a_failed_download_is_reported_and_retried(fake_fastembed, tmp_path):
    model = embedder(fake_fastembed, tmp_path)
    fake_fastembed.download_error = OSError("network unreachable")
    with pytest.raises(EmbedderUnavailable, match="could not be loaded: network unreachable"):
        model.warm()
    fake_fastembed.download_error = None
    assert len(await model.embed_documents(["x"])) == 1
    assert len(fake_fastembed.loads) == 1


def test_missing_fastembed_fails_at_construction(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError, match="mnemotree-mcp"):
        FastEmbedEmbedder("m", source="s", revision="r", dimensions=4, cache_dir=tmp_path)


def test_open_memory_builds_the_fastembed_kind_without_loading(fake_fastembed, tmp_path):
    config = Config(embedding=fake_fastembed.settings(tmp_path / "models"))
    with open_memory(tmp_path / "m.db", config) as engine:
        assert isinstance(engine.embedder, FastEmbedEmbedder)
        assert engine.embedder.spec.provider == "fastembed"
        assert engine.embedder.cache_dir == tmp_path / "models"
    assert fake_fastembed.loads == []


def test_default_settings_describe_the_pinned_model():
    assert DEFAULT_LOCAL_EMBEDDING["kind"] == "fastembed"
    assert DEFAULT_LOCAL_EMBEDDING["dimensions"] == 384
    assert len(DEFAULT_LOCAL_EMBEDDING["revision"]) == 40
    assert DEFAULT_LOCAL_EMBEDDING["cache_dir"] == "~/.mnemotree/models"


@pytest.mark.skipif(
    importlib.util.find_spec("fastembed") is None
    or os.environ.get("MNEMOTREE_TEST_FASTEMBED") != "1",
    reason="downloads the real model: install fastembed and set MNEMOTREE_TEST_FASTEMBED=1",
)
async def test_real_default_model_finds_a_paraphrase():
    settings = dict(DEFAULT_LOCAL_EMBEDDING)
    settings.pop("kind")
    # MNEMOTREE_TEST_FASTEMBED_CACHE keeps a test download out of the user's cache.
    settings["cache_dir"] = os.environ.get("MNEMOTREE_TEST_FASTEMBED_CACHE", settings["cache_dir"])
    model = FastEmbedEmbedder(**settings)
    assert model.files() == TOY_FILES
    near, far = await model.embed_documents(
        ["The spare key is under the blue flower pot.", "Quarterly revenue rose by four percent."]
    )
    query = await model.embed_query("Where did I hide the house key?")
    assert len(query) == 384
    similarity = [sum(a * b for a, b in zip(query, doc, strict=True)) for doc in (near, far)]
    assert similarity[0] > similarity[1]
    assert await asyncio.to_thread(Path(model.cache_dir).is_dir)
