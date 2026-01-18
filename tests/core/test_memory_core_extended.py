"""
Extended tests for core/memory.py to increase coverage from 73.7% to 85%+.

Focuses on untested code paths including:
- remember_async and async ingestion queue
- reflect method for pattern analysis
- forget/delete with cascade
- summarize method
- batch_remember method
- cluster method
- search with filters and different stores
- _queued_memory_stub
- _ensure_ingestion_queue
- _resolve methods for type, importance, tags
- Error handling and edge cases
"""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemotree.analysis.clustering import ClusteringResult
from mnemotree.core.memory import (
    IngestionConfig,
    MemoryCore,
    ModeDefaultsConfig,
    NerConfig,
    RetrievalConfig,
)
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.query import MemoryQuery, MemoryQueryBuilder
from mnemotree.store.base import BaseMemoryStore
from mnemotree.store.protocols import (
    SupportsStructuredQuery,
    SupportsVectorSearch,
)


class MockVectorStore(BaseMemoryStore, SupportsVectorSearch):
    """Mock store with vector search support."""

    async def store_memory(self, memory):
        """Mock implementation - no-op for testing."""

    async def get_memory(self, mid):
        return None

    async def delete_memory(self, mid, *, cascade=False):
        return True

    async def list_memories(self, *, include_embeddings=False):
        return []

    async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
        return []

    async def query_memories(self, query):
        return []

    async def update_connections(self, memory_id, **kwargs):
        # Intentionally empty: Mock method for test fixture
        pass

    async def query_by_entities(self, entities, limit=10):
        return []

    async def close(self):
        """Mock implementation - no-op for testing."""


class MockStructuredQueryStore(BaseMemoryStore, SupportsStructuredQuery):
    """Mock store with structured query support."""

    async def store_memory(self, memory):
        """Mock implementation - no-op for testing."""

    async def get_memory(self, mid):
        return None

    async def delete_memory(self, mid, *, cascade=False):
        return True

    async def list_memories(self, *, include_embeddings=False):
        return []

    async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
        return []

    async def query_memories(self, query):
        return []

    async def update_connections(self, memory_id, **kwargs):
        """Mock implementation - no-op for testing."""
        # Intentionally empty: Mock method for test fixture

    async def query_by_entities(self, entities, limit=10):
        return []

    async def close(self):
        """Mock implementation - no-op for testing."""
        # Intentionally empty: Mock method for test fixture


@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)
    return embeddings


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def basic_memory_core(mock_embeddings):
    """Memory core with minimal configuration."""
    store = MockVectorStore()
    store.store_memory = AsyncMock()
    store.get_memory = AsyncMock(return_value=None)
    store.delete_memory = AsyncMock(return_value=True)

    return MemoryCore(
        store=store,
        embeddings=mock_embeddings,
        mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
        ner_config=NerConfig(enable_ner=False),
    )


@pytest.fixture
def pro_memory_core(mock_llm, mock_embeddings):
    """Memory core with pro features."""
    store = MockStructuredQueryStore()
    store.store_memory = AsyncMock()
    store.query_memories = AsyncMock(return_value=[])
    store.update_connections = AsyncMock()

    return MemoryCore(
        store=store,
        llm=mock_llm,
        embeddings=mock_embeddings,
        mode_defaults=ModeDefaultsConfig(mode="pro", enable_keywords=False),
        ner_config=NerConfig(ner=MagicMock()),
    )


class TestAsyncIngestion:
    """Test async ingestion queue functionality."""

    @pytest.mark.asyncio
    async def test_remember_async_creates_stub_and_queues(self, mock_embeddings):
        """Test that remember_async creates a stub and queues ingestion."""
        store = MockVectorStore()
        store.store_memory = AsyncMock()

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
            ingestion_config=IngestionConfig(async_ingest=True, ingestion_queue_size=50),
        )

        # Remember async should return immediately with a stub
        stub = await memory_core.remember("Async content", importance=0.7)

        assert stub.content == "Async content"
        assert math.isclose(stub.importance, 0.7)
        assert stub.metadata.get("queued") is True
        assert stub.memory_id is not None

        # Give queue time to process
        await asyncio.sleep(0.1)

        # Store should eventually be called
        store.store_memory.assert_called()

    @pytest.mark.asyncio
    async def test_ensure_ingestion_queue_starts_once(self, basic_memory_core):
        """Test that ingestion queue is created only once."""
        basic_memory_core.async_ingest = True

        # First call creates queue
        await basic_memory_core._ensure_ingestion_queue()
        queue1 = basic_memory_core._ingestion_queue
        assert queue1 is not None

        # Second call reuses queue
        await basic_memory_core._ensure_ingestion_queue()
        queue2 = basic_memory_core._ingestion_queue
        assert queue1 is queue2


class TestReflectMethod:
    """Test reflect method for pattern analysis."""

    @pytest.mark.asyncio
    async def test_reflect_without_analyzer_raises_error(self, basic_memory_core):
        """Test that reflect raises error when analyzer not configured."""
        with pytest.raises(RuntimeError, match="Analyzer not configured"):
            await basic_memory_core.reflect()

    @pytest.mark.asyncio
    async def test_reflect_with_no_memories(self, pro_memory_core):
        """Test reflect with no matching memories."""
        pro_memory_core.store.query_memories = AsyncMock(return_value=[])

        result = await pro_memory_core.reflect()

        assert "No memories found" in result["summary"]
        assert result["patterns"] == []
        assert result["insights"] == []

    @pytest.mark.asyncio
    async def test_reflect_analyzes_patterns(self, pro_memory_core):
        """Test reflect analyzes patterns across memories."""
        # Create mock memories
        memories = [
            MemoryItem(
                content="Memory 1",
                memory_type=MemoryType.EPISODIC,
                importance=0.8,
            ),
            MemoryItem(
                content="Memory 2",
                memory_type=MemoryType.SEMANTIC,
                importance=0.9,
            ),
        ]
        pro_memory_core.store.query_memories = AsyncMock(return_value=memories)

        # Mock analyzer
        pro_memory_core.analyzer.analyze_patterns = AsyncMock(
            return_value={"patterns": ["pattern1"], "insights": ["insight1"]}
        )

        result = await pro_memory_core.reflect(min_importance=0.7)

        assert "pattern1" in result["patterns"]
        assert "insight1" in result["insights"]
        pro_memory_core.analyzer.analyze_patterns.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_with_query_builder(self, pro_memory_core):
        """Test reflect with custom query builder."""
        # TODO: Implement full test - currently validates basic call flow
        pro_memory_core.store.query_memories = AsyncMock(return_value=[])

        query_builder = MemoryQueryBuilder().with_tags(["important"])
        await pro_memory_core.reflect(query_builder=query_builder, min_importance=0.5)

        # Verify query was built and used
        pro_memory_core.store.query_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_without_structured_query_support(self, mock_llm, mock_embeddings):
        """Test reflect raises error when store doesn't support structured queries."""

        # Create a store that explicitly does NOT implement SupportsStructuredQuery (no query_memories)
        class MockVectorOnlyStore(BaseMemoryStore, SupportsVectorSearch):
            async def store_memory(self, memory):
                # Intentionally empty: Mock method for inner test class
                pass

            async def get_memory(self, mid):
                return None

            async def delete_memory(self, mid, *, cascade=False):
                return True

            async def list_memories(self, *, include_embeddings=False):
                return []

            async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
                return []

            async def update_connections(self, memory_id, **kwargs):
                # Intentionally empty: Mock method for inner test class
                pass

            async def query_by_entities(self, entities, limit=10):
                return []

            async def close(self):
                # Intentionally empty: Mock method for inner test class
                pass

        store = MockVectorOnlyStore()  # Doesn't support SupportsStructuredQuery

        memory_core = MemoryCore(
            store=store,
            llm=mock_llm,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="pro", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),  # Disable NER to avoid model loading
        )

        with pytest.raises(
            NotImplementedError,
            match="(does not support structured queries|Structured queries are not supported)",
        ):
            await memory_core.reflect()


class TestForgetMethod:
    """Test forget/delete functionality."""

    @pytest.mark.asyncio
    async def test_forget_calls_persistence_delete(self, basic_memory_core):
        """Test forget delegates to persistence layer."""
        basic_memory_core.store.delete_memory = AsyncMock(return_value=True)

        result = await basic_memory_core.forget("mem-123")

        assert result is True
        basic_memory_core.store.delete_memory.assert_called_once_with("mem-123", cascade=False)

    @pytest.mark.asyncio
    async def test_forget_with_cascade(self, basic_memory_core):
        """Test forget with cascade option."""
        basic_memory_core.store.delete_memory = AsyncMock(return_value=True)

        await basic_memory_core.forget("mem-123", cascade=True)

        basic_memory_core.store.delete_memory.assert_called_once_with("mem-123", cascade=True)


class TestSummarizeMethod:
    """Test summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_without_summarizer_raises_error(self, basic_memory_core):
        """Test summarize raises error when summarizer not configured."""
        memories = [MemoryItem(content="test", memory_type=MemoryType.SEMANTIC, importance=0.5)]

        with pytest.raises(RuntimeError, match="Summarizer not configured"):
            await basic_memory_core.summarize(memories)

    @pytest.mark.asyncio
    async def test_summarize_empty_list_returns_empty_string(self, pro_memory_core):
        """Test summarize with empty list returns empty string."""
        result = await pro_memory_core.summarize([])

        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_multiple_memories(self, pro_memory_core):
        """Test summarize generates summary for multiple memories."""
        memories = [
            MemoryItem(content="Memory 1", memory_type=MemoryType.EPISODIC, importance=0.8),
            MemoryItem(content="Memory 2", memory_type=MemoryType.SEMANTIC, importance=0.7),
        ]

        pro_memory_core.summarizer.summarize = AsyncMock(return_value="Combined summary")

        result = await pro_memory_core.summarize(memories, format="text")

        assert result == "Combined summary"
        pro_memory_core.summarizer.summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_structured_format(self, pro_memory_core):
        """Test summarize with structured format."""
        memories = [MemoryItem(content="test", memory_type=MemoryType.SEMANTIC, importance=0.5)]

        pro_memory_core.summarizer.summarize = AsyncMock(
            return_value={"summary": "text", "key_points": ["point1"]}
        )

        result = await pro_memory_core.summarize(memories, format="structured")

        assert isinstance(result, dict)
        assert "summary" in result


class TestBatchRemember:
    """Test batch_remember method."""

    @pytest.mark.asyncio
    async def test_batch_remember_stores_multiple_memories(self, basic_memory_core):
        """Test batch_remember creates multiple memories."""
        contents = ["Memory 1", "Memory 2", "Memory 3"]

        memories = await basic_memory_core.batch_remember(contents, analyze=False)

        assert len(memories) == 3
        assert [m.content for m in memories] == contents

    @pytest.mark.asyncio
    async def test_batch_remember_with_shared_context(self, basic_memory_core):
        """Test batch_remember with shared context."""
        # TODO: Expand test with more context validation scenarios
        contents = ["First", "Second"]
        context = {"source": "batch_import"}

        memories = await basic_memory_core.batch_remember(contents, analyze=False, context=context)

        assert all(m.context == context for m in memories)


class TestSearchMethod:
    """Test search method with various store types."""

    @pytest.mark.asyncio
    async def test_search_with_filters_requires_structured_query_store(self, mock_embeddings):
        """Test search with filters on non-structured store raises error."""

        # Create a store that explicitly does NOT implement SupportsStructuredQuery
        class MockVectorOnlyStore(BaseMemoryStore, SupportsVectorSearch):
            async def store_memory(self, memory):
                # Intentionally empty: Mock method for test
                pass

            async def get_memory(self, mid):
                return None

            async def delete_memory(self, mid, *, cascade=False):
                return True

            async def list_memories(self, *, include_embeddings=False):
                return []

            async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
                return []

            async def update_connections(self, memory_id, **kwargs):
                # Intentionally empty: Mock method for test
                pass

            async def query_by_entities(self, entities, limit=10):
                return []

            async def close(self):
                # Intentionally empty: Mock method for test
                pass

        store = MockVectorOnlyStore()  # Only supports vector search

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )

        with pytest.raises(
            NotImplementedError,
            match="(does not support structured queries|Structured queries are not supported)",
        ):
            await memory_core.search("query", filters={"memory_type": "episodic"})

    @pytest.mark.asyncio
    async def test_search_uses_structured_query_with_filters(self, mock_embeddings):
        """Test search uses structured query when filters provided."""
        store = MockStructuredQueryStore()
        store.query_memories = AsyncMock(return_value=[])

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )

        await memory_core.search("query", filters={"tag": "important"}, limit=5)

        store.query_memories.assert_called_once()
        call_args = store.query_memories.call_args[0][0]
        assert isinstance(call_args, MemoryQuery)
        assert len(call_args.filters) == 1

    @pytest.mark.asyncio
    async def test_search_falls_back_to_vector_search(self, mock_embeddings):
        """Test search uses vector search when no filters and store supports it."""

        # Use a pure object that does NOT inherit from BaseMemoryStore to avoid
        # accidentally matching SupportsStructuredQuery protocol via inherited methods
        class PureVectorStore:
            async def store_memory(self, memory):
                # Intentionally empty: Mock method for test
                pass

            async def get_memory(self, mid):
                return None

            async def delete_memory(self, mid, *, cascade=False):
                return True

            async def list_memories(self, *, include_embeddings=False):
                return []

            async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
                return []

            async def close(self):
                # Intentionally empty: Mock method for test
                pass

        store = PureVectorStore()
        store.get_similar_memories = AsyncMock(return_value=[])

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )

        await memory_core.search("query", limit=10)

        store.get_similar_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_without_vector_support_raises_error(self, mock_embeddings):
        """Test search raises error when store doesn't support any search method."""

        class ConcreteBaseStore(BaseMemoryStore):
            async def store_memory(self, memory):
                # Intentionally empty: Mock method for test
                pass

            async def get_memory(self, mid):
                return None

            async def delete_memory(self, mid, *, cascade=False):
                return True

            async def close(self):
                # Intentionally empty: Mock method for test
                pass

        store = ConcreteBaseStore()

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )

        # When store inherits from BaseMemoryStore, it technically satisfies the protocols checks
        # because the methods exist (raising NotImplementedError).
        # MemoryCore prefers structured query if supported.
        with pytest.raises(
            NotImplementedError,
            match="(Structured queries are not supported|Vector similarity search is not supported)",
        ):
            await memory_core.search("query")

    @pytest.mark.asyncio
    async def test_search_with_bm25_empty_index(self, mock_embeddings):
        """Test search with BM25 but empty index falls back to vector search."""

        # Use a pure object to avoid protocol confusion
        class PureVectorStore:
            async def store_memory(self, memory):
                # Intentionally empty: Mock method for test
                pass

            async def get_memory(self, mid):
                return None

            async def delete_memory(self, mid, *, cascade=False):
                return True

            async def list_memories(self, *, include_embeddings=False):
                return []

            async def get_similar_memories(self, query, query_embedding, top_k=5, filters=None):
                return []

            async def close(self):
                # Intentionally empty: Mock method for test
                pass

        store = PureVectorStore()
        store.get_similar_memories = AsyncMock(return_value=[])

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
            retrieval_config=RetrievalConfig(enable_bm25=True),
        )

        # BM25 index is empty (doc_count = 0)
        assert memory_core.index_manager.doc_count == 0

        await memory_core.search("query")

        # Should fall back to vector search
        store.get_similar_memories.assert_called_once()


class TestClusterMethod:
    """Test cluster method."""

    @pytest.mark.asyncio
    async def test_cluster_without_clusterer_raises_error(self, basic_memory_core):
        """Test cluster raises error when clusterer not configured."""
        query_builder = MemoryQueryBuilder()

        with pytest.raises(RuntimeError, match="Clusterer not configured"):
            await basic_memory_core.cluster(query=query_builder)

    @pytest.mark.asyncio
    async def test_cluster_without_query_raises_error(self, pro_memory_core):
        """Test cluster without query raises ValueError."""
        with pytest.raises(ValueError, match="Query is required"):
            await pro_memory_core.cluster(query=None)

    @pytest.mark.asyncio
    async def test_cluster_returns_memories_and_results(self, pro_memory_core):
        """Test cluster returns both memories and clustering results."""
        memories = [
            MemoryItem(
                content="Memory 1",
                memory_type=MemoryType.SEMANTIC,
                importance=0.8,
                embedding=[0.1] * 768,
            ),
            MemoryItem(
                content="Memory 2",
                memory_type=MemoryType.SEMANTIC,
                importance=0.7,
                embedding=[0.2] * 768,
            ),
        ]

        # Mock recall to return memories
        pro_memory_core.retrieval.recall = AsyncMock(return_value=memories)

        # Mock clusterer
        clustering_result = ClusteringResult(
            cluster_ids=[0, 0],
            centroids=None,
            cluster_sizes={0: 2},
            cluster_summaries={0: "Summary"},
        )
        pro_memory_core.clusterer.cluster_memories = AsyncMock(return_value=clustering_result)

        query = MemoryQueryBuilder().importance_range(min_value=0.5)
        returned_memories, results = await pro_memory_core.cluster(query=query)

        assert len(returned_memories) == 2
        assert len(results.cluster_ids) == 2
        assert results.cluster_ids == [0, 0]


class TestResolverMethods:
    """Test internal resolver methods."""

    def test_resolve_importance_and_type_with_analysis(self, basic_memory_core):
        """Test resolution uses analysis when provided."""
        from mnemotree.analysis.models import MemoryAnalysisResult

        analysis = MemoryAnalysisResult(
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            tags=["important"],
            emotions=[],
            emotional_valence=0.5,
            emotional_arousal=0.5,
            linked_concepts=[],
            context_summary=None,
        )

        mem_type, importance = basic_memory_core._resolve_importance_and_type(
            memory_type=None,
            importance=None,
            analysis=analysis,
        )

        assert mem_type == MemoryType.EPISODIC
        assert math.isclose(importance, 0.9)

    def test_resolve_importance_and_type_overrides_analysis(self, basic_memory_core):
        """Test that explicit values override analysis."""
        from mnemotree.analysis.models import MemoryAnalysisResult

        analysis = MemoryAnalysisResult(
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            tags=[],
            emotions=[],
            emotional_valence=0.5,
            emotional_arousal=0.5,
            linked_concepts=[],
            context_summary=None,
        )

        mem_type, importance = basic_memory_core._resolve_importance_and_type(
            memory_type=MemoryType.PROCEDURAL,
            importance=0.3,
            analysis=analysis,
        )

        assert mem_type == MemoryType.PROCEDURAL
        assert math.isclose(importance, 0.3)

    def test_resolve_importance_and_type_without_analysis(self, basic_memory_core):
        """Test resolution uses defaults when no analysis."""
        mem_type, importance = basic_memory_core._resolve_importance_and_type(
            memory_type=None,
            importance=None,
            analysis=None,
        )

        assert mem_type == MemoryType.SEMANTIC
        assert math.isclose(importance, 0.5)  # default

    def test_resolve_tags_combines_all_sources(self, basic_memory_core):
        """Test tag resolution combines tags from all sources."""
        from mnemotree.analysis.models import MemoryAnalysisResult

        analysis = MemoryAnalysisResult(
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            tags=["analysis_tag"],
            emotions=[],
            emotional_valence=None,
            emotional_arousal=None,
            linked_concepts=[],
            context_summary=None,
        )

        tags = basic_memory_core._resolve_tags(
            tags=["user_tag"],
            analysis=analysis,
            keyword_tags=["keyword_tag"],
        )

        assert set(tags) == {"user_tag", "analysis_tag", "keyword_tag"}

    def test_resolve_tags_deduplicates(self, basic_memory_core):
        """Test tag resolution removes duplicates."""
        from mnemotree.analysis.models import MemoryAnalysisResult

        analysis = MemoryAnalysisResult(
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            tags=["duplicate", "unique"],
            emotions=[],
            emotional_valence=None,
            emotional_arousal=None,
            linked_concepts=[],
            context_summary=None,
        )

        tags = basic_memory_core._resolve_tags(
            tags=["duplicate"],
            analysis=analysis,
            keyword_tags=["duplicate", "keyword"],
        )

        # Should contain each tag only once
        assert len(tags) == 3
        assert set(tags) == {"duplicate", "unique", "keyword"}


class TestGetEmbedding:
    """Test get_embedding method."""

    @pytest.mark.asyncio
    async def test_get_embedding_returns_vector(self, basic_memory_core):
        """Test get_embedding returns embedding vector."""
        embedding = await basic_memory_core.get_embedding("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # Mock embedding size

    @pytest.mark.asyncio
    async def test_get_embedding_without_embedder_raises_error(self):
        """Test get_embedding raises error when embedder not configured."""
        store = MockVectorStore()

        # Use patch to prevent actual instantiation of LocalSentenceTransformerEmbeddings
        # which would fail if dependencies are missing, even though we intend to trigger
        # "Embedder not configured" error later.
        with patch("mnemotree.core.memory.LocalSentenceTransformerEmbeddings"):
            memory_core = MemoryCore(
                store=store,
                embeddings=None,
                mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
                ner_config=NerConfig(enable_ner=False),
            )

        memory_core.embedder = None  # Explicitly set to None

        with pytest.raises(RuntimeError, match="Embedder not configured"):
            await memory_core.get_embedding("test")


class TestDecayAndReinforce:
    """Test decay_and_reinforce method."""

    def test_decay_and_reinforce_returns_none(self, basic_memory_core):
        """Test decay_and_reinforce currently returns None (stub)."""
        result = basic_memory_core.decay_and_reinforce()
        assert result is None


class TestModeConfiguration:
    """Test mode configuration and defaults."""

    def test_lite_mode_disables_analysis_by_default(self, mock_embeddings):
        """Test lite mode doesn't enable analysis/summarization by default."""
        store = MockVectorStore()

        memory_core = MemoryCore(
            store=store,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="lite", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )

        assert memory_core.default_analyze is False
        assert memory_core.default_summarize is False

    def test_pro_mode_enables_analysis_by_default(self, mock_llm, mock_embeddings):
        """Test pro mode enables analysis/summarization by default."""
        store = MockVectorStore()

        memory_core = MemoryCore(
            store=store,
            llm=mock_llm,
            embeddings=mock_embeddings,
            mode_defaults=ModeDefaultsConfig(mode="pro", enable_keywords=False),
            ner_config=NerConfig(enable_ner=False),
        )

        assert memory_core.default_analyze is True
        assert memory_core.default_summarize is True

    @pytest.mark.asyncio
    async def test_pro_mode_with_env_vars_creates_llm(self, mock_embeddings):
        """Test pro mode creates LLM from environment variables."""
        # TODO: Mock LLM creation to avoid external dependencies
        store = MockVectorStore()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            memory_core = MemoryCore(
                store=store,
                embeddings=mock_embeddings,
                mode_defaults=ModeDefaultsConfig(mode="pro", enable_keywords=False),
                ner_config=NerConfig(enable_ner=False),
            )

            # Should have created analyzer and summarizer
            assert memory_core.analyzer is not None
            assert memory_core.summarizer is not None
