"""The 1.0 module paths under mnemotree.evidence keep working after the move."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mnemotree
import mnemotree.evidence

MOVED = {
    "models": "mnemotree.core.models",
    "protocols": "mnemotree.core.protocols",
    "temporal": "mnemotree.core.temporal",
    "chunking": "mnemotree.ingest.chunking",
    "atoms": "mnemotree.ingest.atoms",
    "ranking": "mnemotree.retrieval.ranking",
    "packing": "mnemotree.retrieval.packing",
    "store": "mnemotree.storage.store",
    "schema": "mnemotree.storage.schema",
    "vectors": "mnemotree.storage.vectors",
    "lexical": "mnemotree.storage.lexical",
    "atom_store": "mnemotree.storage.atom_store",
    "providers": "mnemotree.providers.embeddings",
    "retry": "mnemotree.providers.retry",
    "server": "mnemotree.service.app",
    "engine": "mnemotree.engine",
    "evaluation": "mnemotree.evaluation",
    "cli": "mnemotree.cli.evidence",
}
SRC = str(Path(mnemotree.__file__).parents[1])


@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_old_path_is_the_moved_module(old, new):
    module = importlib.import_module(f"mnemotree.evidence.{old}")
    assert module is importlib.import_module(new)
    assert getattr(mnemotree.evidence, old) is module


def test_names_used_from_the_old_paths_still_import():
    from mnemotree.evidence.atoms import OpenAIExtractor
    from mnemotree.evidence.cli import Config, main, open_memory, secret
    from mnemotree.evidence.engine import QUERY_EMBED_TOKENS, Memory, fuse, model_query
    from mnemotree.evidence.models import Evidence, SearchOptions, render_line
    from mnemotree.evidence.packing import TiktokenCounter
    from mnemotree.evidence.providers import OpenAIEmbedder, SentenceTransformerEmbedder
    from mnemotree.evidence.server import aml_options, create_app
    from mnemotree.evidence.store import CHUNK_FTS, IngestConflict, vector_blob

    assert Memory is mnemotree.evidence.Memory
    assert IngestConflict is mnemotree.IngestConflict
    for name in (OpenAIExtractor, Config, main, open_memory, secret, QUERY_EMBED_TOKENS, fuse):
        assert name is not None
    for name in (model_query, Evidence, SearchOptions, render_line, TiktokenCounter):
        assert name is not None
    for name in (OpenAIEmbedder, SentenceTransformerEmbedder, aml_options, create_app):
        assert name is not None
    assert CHUNK_FTS and vector_blob


def test_patching_through_an_old_path_reaches_the_moved_module(monkeypatch):
    from mnemotree.evidence import store as old
    from mnemotree.storage import store as new

    monkeypatch.setattr(old, "now", lambda: "patched")
    assert new.now() == "patched"


def run(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )


def test_importing_the_package_loads_no_optional_extra():
    probe = run(
        "import sys, mnemotree, mnemotree.evidence, mnemotree.mcp.server\n"
        "loaded = [m for m in ('fastapi', 'uvicorn', 'httpx', 'mnemotree.service.app')"
        " if m in sys.modules]\n"
        "assert not loaded, loaded"
    )
    assert probe.returncode == 0, probe.stderr


def test_old_cli_module_still_runs_as_a_script():
    env = {**os.environ, "PYTHONPATH": SRC}
    result = subprocess.run(
        [sys.executable, "-m", "mnemotree.evidence.cli", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "serve" in result.stdout
