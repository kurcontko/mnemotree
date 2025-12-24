
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from mnemotree.core.memory import MemoryCore
from mnemotree.core.models import MemoryItem
from mnemotree.store.base import BaseMemoryStore
from mnemotree.core.models import MemoryType

class MockStore(BaseMemoryStore):
    async def store_memory(self, memory): pass
    async def get_memory(self, mid): return None
    async def delete_memory(self, mid, cascade=False): return True
    async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None): return []
    async def query_memories(self, query): return []
    async def update_connections(self, memory_id, **kwargs): pass
    async def query_by_entities(self, entities): return []
    async def close(self): pass

@pytest.fixture
def mock_store():
    store = MockStore()
    store.store_memory = AsyncMock()
    store.update_connections = AsyncMock()
    return store

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mocking behaviors if needed
    return llm

@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    # Mock async embedding call
    embeddings.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    return embeddings

@pytest.fixture
def memory_core(mock_store, mock_llm, mock_embeddings):
    return MemoryCore(
        store=mock_store,
        llm=mock_llm,
        embeddings=mock_embeddings,
        ner=MagicMock()
    )

@pytest.mark.asyncio
async def test_remember_flow(memory_core, mock_store):
    # Setup mocks to avoid actual analysis
    memory_core.analyzer.analyze = AsyncMock(return_value=MagicMock(
        memory_type=MemoryType.EPISODIC,
        importance=0.8,
        tags=["test"],
        emotions=["joy"],
        linked_concepts=[]
    ))
    memory_core.summarizer.summarize = AsyncMock(return_value="Summary")
    memory_core.ner.extract_entities = AsyncMock(return_value=MagicMock(entities={}, mentions={}))

    memory = await memory_core.remember("Test memory content")
    
    assert memory.content == "Test memory content"
    assert memory.summary == "Summary"
    
    # Verify store was called
    mock_store.store_memory.assert_called_once()
    
@pytest.mark.asyncio
async def test_remember_with_references(memory_core, mock_store):
    # Setup mocks
    memory_core.analyzer.analyze = AsyncMock(return_value=MagicMock(
        memory_type=MemoryType.SEMANTIC, importance=0.5, tags=[], emotions=[], linked_concepts=[]
    ))
    memory_core.summarizer.summarize = AsyncMock(return_value="Summary")
    memory_core.ner.extract_entities = AsyncMock(return_value=MagicMock(entities={}, mentions={}))

    await memory_core.remember(
        "Test content", 
        references=["prev-id"]
    )
    
    # Store called
    mock_store.store_memory.assert_called_once()
    # Connect called
    mock_store.update_connections.assert_called_once()

@pytest.mark.asyncio
async def test_connect(memory_core, mock_store):
    await memory_core.connect("mem-1", related_to=["mem-2"])
    mock_store.update_connections.assert_called_with(
        memory_id="mem-1",
        related_ids=["mem-2"],
        conflict_ids=None,
        previous_id=None,
        next_id=None
    )
