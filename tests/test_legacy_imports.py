"""Fresh interpreters catch store/core import cycles hidden by pytest collection order."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("store_module", ["sqlite_vec_store", "neo4j_store"])
def test_store_import_first_preserves_core_public_exports(store_module):
    if store_module == "neo4j_store" and importlib.util.find_spec("neo4j") is None:
        pytest.skip("optional Neo4j driver is not installed")
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
    script = f"""
from importlib import import_module
import_module('mnemotree.store.{store_module}')
import mnemotree.core as core
expected = {{
    'MemoryCore', 'MemoryCoreBuilder', 'MemoryMode', 'MemoryItem', 'MemoryLink',
    'MemoryQuery', 'MemoryQueryBuilder', 'MemoryType', 'LinkType', 'RememberOptions',
    'RecallFilters', 'RecallOptions', 'IngestionConfig', 'ModeDefaultsConfig',
    'NerConfig', 'RetrievalConfig', 'RetrievalMode', 'ScoringConfig', 'HybridRetriever',
    'CrossEncoderReranker', 'NoOpReranker', 'FusionStrategy', 'RetrievalStage',
    'RetrieverFactory',
}}
assert set(core.__all__) == expected
assert expected <= set(dir(core))
for name in core.__all__:
    value = getattr(core, name)
    module = import_module(core._EXPORTS[name], core.__name__)
    assert value is getattr(module, name)
    assert getattr(core, name) is value
from mnemotree import MemoryCore
assert MemoryCore is core.MemoryCore
try:
    core.missing_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown exports must raise AttributeError')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
