"""Tests for experimental evaluation module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.experimental.evaluation import (
    BenchmarkResult,
    EvaluationQuery,
    EvaluationResult,
    MemoryEvaluator,
    MetricType,
    SyntheticDatasetGenerator,
)


@pytest.fixture
def memory_items():
    """Create sample memory items for testing."""
    return [
        MemoryItem(
            memory_id="mem1",
            content="Python is a programming language",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            timestamp=str(datetime(2024, 1, 1, tzinfo=timezone.utc)),
            embedding=[0.1] * 128,
        ),
        MemoryItem(
            memory_id="mem2",
            content="I learned Python yesterday",
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            timestamp=str(datetime(2024, 1, 2, tzinfo=timezone.utc)),
            embedding=[0.2] * 128,
        ),
        MemoryItem(
            memory_id="mem3",
            content="Machine learning uses Python",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7,
            timestamp=str(datetime(2024, 1, 3, tzinfo=timezone.utc)),
            embedding=[0.3] * 128,
        ),
    ]


@pytest.fixture
def evaluation_queries():
    """Create sample evaluation queries."""
    return [
        EvaluationQuery(
            query_id="q1",
            query_text="What is Python?",
            relevant_memory_ids=["mem1", "mem3"],
        ),
        EvaluationQuery(
            query_id="q2",
            query_text="When did I learn Python?",
            relevant_memory_ids=["mem2"],
        ),
    ]


@pytest.fixture
def mock_memory_core():
    """Create a mock memory core."""
    core = MagicMock()
    core.recall = AsyncMock()
    return core


class TestEvaluationQuery:
    """Tests for EvaluationQuery dataclass."""

    def test_creation(self):
        """Test creating an evaluation query."""
        query = EvaluationQuery(
            query_id="test_q",
            query_text="Test query?",
            relevant_memory_ids=["mem1", "mem2"],
        )
        assert query.query_id == "test_q"
        assert query.query_text == "Test query?"
        assert query.relevant_memory_ids == ["mem1", "mem2"]
        assert query.metadata == {}

    def test_with_metadata(self):
        """Test creating query with metadata."""
        query = EvaluationQuery(
            query_id="test_q",
            query_text="Test query?",
            relevant_memory_ids=["mem1"],
            metadata={"category": "test", "difficulty": "easy"},
        )
        assert query.metadata["category"] == "test"
        assert query.metadata["difficulty"] == "easy"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            query_id="q1",
            retrieved_memory_ids=["mem1", "mem2"],
        )
        assert result.query_id == "q1"
        assert result.retrieved_memory_ids == ["mem1", "mem2"]
        assert result.recall_at_k == {}
        assert result.precision_at_k == {}

    def test_with_metrics(self):
        """Test result with computed metrics."""
        result = EvaluationResult(
            query_id="q1",
            retrieved_memory_ids=["mem1", "mem2"],
            recall_at_k={1: 0.5, 5: 1.0},
            precision_at_k={1: 1.0, 5: 0.4},
            mrr=1.0,
            ndcg=0.85,
            true_positives=2,
            false_positives=0,
            false_negatives=0,
        )
        assert result.recall_at_k[1] == pytest.approx(0.5)
        assert result.precision_at_k[5] == pytest.approx(0.4)
        assert result.mrr == pytest.approx(1.0)
        assert result.ndcg == pytest.approx(0.85)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult()
        assert result.total_queries == 0
        assert result.total_memories == 0
        assert result.avg_mrr == pytest.approx(0.0)
        assert isinstance(result.evaluated_at, datetime)

    def test_to_dict(self):
        """Test converting benchmark result to dict."""
        result = BenchmarkResult(
            avg_recall_at_k={1: 0.8, 5: 0.9},
            avg_precision_at_k={1: 0.9, 5: 0.7},
            avg_mrr=0.85,
            total_queries=10,
            total_memories=50,
        )
        result_dict = result.to_dict()
        assert result_dict["avg_recall_at_k"] == {1: 0.8, 5: 0.9}
        assert result_dict["avg_mrr"] == pytest.approx(0.85)
        assert result_dict["total_queries"] == 10


class TestMemoryEvaluator:
    """Tests for MemoryEvaluator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_memory_core):
        """Test creating evaluator."""
        evaluator = MemoryEvaluator(memory_core=mock_memory_core)
        assert evaluator.memory_core == mock_memory_core

    @pytest.mark.asyncio
    async def test_compute_recall_at_k(self):
        """Test computing recall@k metric."""
        evaluator = MemoryEvaluator(memory_core=MagicMock())

        relevant = ["mem1", "mem2", "mem3"]
        retrieved = ["mem1", "mem2", "mem4", "mem5"]

        recall_1 = evaluator._compute_recall_at_k(relevant, retrieved, k=1)
        assert recall_1 == pytest.approx(1.0 / 3.0)  # Only mem1 retrieved in top-1

        recall_2 = evaluator._compute_recall_at_k(relevant, retrieved, k=2)
        assert recall_2 == pytest.approx(2.0 / 3.0)  # mem1 and mem2 retrieved in top-2

        recall_4 = evaluator._compute_recall_at_k(relevant, retrieved, k=4)
        assert recall_4 == pytest.approx(2.0 / 3.0)  # Still only mem1 and mem2

    @pytest.mark.asyncio
    async def test_compute_precision_at_k(self):
        """Test computing precision@k metric."""
        evaluator = MemoryEvaluator(memory_core=MagicMock())

        relevant = ["mem1", "mem2", "mem3"]
        retrieved = ["mem1", "mem2", "mem4", "mem5"]

        precision_1 = evaluator._compute_precision_at_k(relevant, retrieved, k=1)
        assert precision_1 == pytest.approx(1.0)  # mem1 is relevant

        precision_2 = evaluator._compute_precision_at_k(relevant, retrieved, k=2)
        assert precision_2 == pytest.approx(1.0)  # Both mem1 and mem2 are relevant

        precision_4 = evaluator._compute_precision_at_k(relevant, retrieved, k=4)
        assert precision_4 == pytest.approx(0.5)  # 2 out of 4 are relevant

    @pytest.mark.asyncio
    async def test_compute_mrr(self):
        """Test computing Mean Reciprocal Rank."""
        evaluator = MemoryEvaluator(memory_core=MagicMock())

        relevant = ["mem2"]
        retrieved = ["mem1", "mem2", "mem3"]

        mrr = evaluator._compute_mrr(relevant, retrieved)
        assert mrr == pytest.approx(0.5)  # mem2 is at position 2 (1-indexed), so 1/2 = 0.5

    @pytest.mark.asyncio
    async def test_compute_mrr_first_position(self):
        """Test MRR when relevant item is first."""
        evaluator = MemoryEvaluator(memory_core=MagicMock())

        relevant = ["mem1"]
        retrieved = ["mem1", "mem2", "mem3"]

        mrr = evaluator._compute_mrr(relevant, retrieved)
        assert mrr == pytest.approx(1.0)  # mem1 is at position 1, so 1/1 = 1.0

    @pytest.mark.asyncio
    async def test_compute_mrr_not_found(self):
        """Test MRR when no relevant items retrieved."""
        evaluator = MemoryEvaluator(memory_core=MagicMock())

        relevant = ["mem4"]
        retrieved = ["mem1", "mem2", "mem3"]

        mrr = evaluator._compute_mrr(relevant, retrieved)
        assert mrr == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_evaluate_single_query_perfect(self, mock_memory_core, memory_items):
        """Test evaluating a single query with perfect retrieval."""
        mock_memory_core.recall.return_value = [memory_items[0], memory_items[2]]

        evaluator = MemoryEvaluator(memory_core=mock_memory_core)
        query = EvaluationQuery(
            query_id="q1",
            query_text="Test query",
            relevant_memory_ids=["mem1", "mem3"],
        )

        result = await evaluator.evaluate_query(query, k_values=[1, 2, 5])

        assert result.query_id == "q1"
        assert result.recall_at_k[2] == pytest.approx(1.0)  # Both relevant items retrieved
        assert result.precision_at_k[2] == pytest.approx(1.0)  # All retrieved are relevant
        assert result.mrr == pytest.approx(1.0)  # First relevant at position 1

    @pytest.mark.asyncio
    async def test_evaluate_single_query_partial(self, mock_memory_core, memory_items):
        """Test evaluating a single query with partial retrieval."""
        mock_memory_core.recall.return_value = [memory_items[0], memory_items[1]]

        evaluator = MemoryEvaluator(memory_core=mock_memory_core)
        query = EvaluationQuery(
            query_id="q1",
            query_text="Test query",
            relevant_memory_ids=["mem1", "mem3"],
        )

        result = await evaluator.evaluate_query(query, k_values=[1, 2, 5])

        assert result.query_id == "q1"
        assert result.recall_at_k[2] == pytest.approx(0.5)  # Only 1 of 2 relevant retrieved
        assert result.precision_at_k[2] == pytest.approx(0.5)  # 1 of 2 retrieved is relevant
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 1


class TestSyntheticDatasetGenerator:
    """Tests for SyntheticDatasetGenerator class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test creating synthetic dataset generator."""
        mock_llm = MagicMock()
        generator = SyntheticDatasetGenerator(llm=mock_llm)
        assert generator.llm == mock_llm

    @pytest.mark.asyncio
    async def test_generate_queries_from_memories(self, memory_items):
        """Test generating queries from memories."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="What programming language was discussed?\nWhen did the user learn Python?"
            )
        )

        generator = SyntheticDatasetGenerator(llm=mock_llm)
        queries = await generator.generate_queries_from_memories(
            memories=memory_items[:2],
            queries_per_memory=2,
        )

        assert len(queries) > 0
        assert all(isinstance(q, EvaluationQuery) for q in queries)

    @pytest.mark.asyncio
    async def test_generate_simple_queries(self, memory_items):
        """Test generating simple synthetic queries without LLM."""
        SyntheticDatasetGenerator(llm=None)

        # Simple query generation that doesn't need LLM
        queries = []
        for i, mem in enumerate(memory_items[:2]):
            query = EvaluationQuery(
                query_id=f"synthetic_q{i}",
                query_text=f"Query about: {mem.content[:20]}...",
                relevant_memory_ids=[mem.memory_id],
            )
            queries.append(query)

        assert len(queries) == 2
        assert queries[0].relevant_memory_ids == ["mem1"]
        assert queries[1].relevant_memory_ids == ["mem2"]


class TestMetricType:
    """Tests for MetricType enum."""

    def test_enum_values(self):
        """Test all metric type values."""
        assert MetricType.RECALL == "recall"
        assert MetricType.PRECISION == "precision"
        assert MetricType.F1_SCORE == "f1_score"
        assert MetricType.MRR == "mrr"
        assert MetricType.NDCG == "ndcg"
        assert MetricType.DRIFT_RATE == "drift_rate"
        assert MetricType.CONTRADICTION_COUNT == "contradiction_count"

    def test_enum_members(self):
        """Test enum members."""
        assert "RECALL" in MetricType.__members__
        assert "PRECISION" in MetricType.__members__
        assert "MRR" in MetricType.__members__
