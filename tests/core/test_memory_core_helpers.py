from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core.memory import MemoryCore, ModeDefaultsConfig, NerConfig
from mnemotree.core.models import MemoryType
from mnemotree.store.base import BaseMemoryStore


class HelperStore(BaseMemoryStore):
    def __init__(self):
        self.storage: dict[str, object] = {}

    async def store_memory(self, memory):
        self.storage[memory.memory_id] = memory

    async def get_memory(self, memory_id):
        return self.storage.get(memory_id)

    async def delete_memory(self, memory_id, *, cascade: bool = False):
        return self.storage.pop(memory_id, None) is not None

    async def close(self):
        self.storage.clear()


@pytest.fixture
def helper_store():
    return HelperStore()


@pytest.fixture
def stub_embeddings():
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.0, 0.0])
    return embeddings


@pytest.fixture
def helper_core(helper_store, stub_embeddings):
    return MemoryCore(
        store=helper_store,
        embeddings=stub_embeddings,
        ner_config=NerConfig(enable_ner=False),
        mode_defaults=ModeDefaultsConfig(enable_keywords=False),
    )


class StubKeywordExtractor:
    async def extract(self, text: str):
        return ["kw"]


def test_mode_config_respects_overrides(helper_store, stub_embeddings):
    pro_core = MemoryCore(
        store=helper_store,
        embeddings=stub_embeddings,
        mode_defaults=ModeDefaultsConfig(
            mode="pro",
            default_analyze=False,
            default_summarize=False,
        ),
        ner_config=NerConfig(enable_ner=False),
    )
    assert pro_core.default_analyze is False
    assert pro_core.default_summarize is False

    stub_kw = StubKeywordExtractor()
    lite_core = MemoryCore(
        store=helper_store,
        embeddings=stub_embeddings,
        mode_defaults=ModeDefaultsConfig(
            mode="lite",
            enable_keywords=False,
            keyword_extractor=stub_kw,
        ),
        ner_config=NerConfig(enable_ner=False),
    )
    assert lite_core.enable_keywords is True
    assert lite_core.keyword_extractor is stub_kw


def test_resolve_importance_and_type_prefers_analysis(helper_core):
    analysis = SimpleNamespace(memory_type=MemoryType.EPISODIC, importance=0.9)
    resolved_type, resolved_importance = helper_core._resolve_importance_and_type(
        memory_type=None,
        importance=None,
        analysis=analysis,
    )
    assert resolved_type == MemoryType.EPISODIC
    assert resolved_importance == pytest.approx(0.9)

    explicit_type = MemoryType.PROCEDURAL
    fallback_type, fallback_importance = helper_core._resolve_importance_and_type(
        memory_type=explicit_type,
        importance=0.2,
        analysis=None,
    )
    assert fallback_type == explicit_type
    assert fallback_importance == pytest.approx(0.2)


def test_resolve_tags_merges_manual_analysis_and_keywords(helper_core):
    analysis = SimpleNamespace(tags=["analysis"])
    merged = helper_core._resolve_tags(
        tags=["manual"],
        analysis=analysis,
        keyword_tags=["kw"],
    )
    assert set(merged) == {"manual", "analysis", "kw"}
