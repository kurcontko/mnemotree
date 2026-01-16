from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:  # type: ignore[misc]
        return False

DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_MODEL = "gpt-4.1-mini"
METRIC_PRECISION = "precision@k"
METRIC_RECALL = "recall@k"
METRIC_NDCG = "ndcg@k"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MemoryRecord:
    content: str
    memory_type: Optional[str] = None
    concepts: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)
    emotion: Optional[str] = None
    importance: Optional[float] = None


@dataclass
class QueryCase:
    query: str
    expected_memories: List[str] = field(default_factory=list)
    expected_ids: List[str] = field(default_factory=list)
    expected_answer: Optional[str] = None
    description: str = ""
    query_type: Optional[str] = None


@dataclass
class EvalConfig:
    data_dir: Path
    memories_file: str
    queries_file: str
    k_values: List[int]
    store_type: str
    mode: str
    scoring: bool
    enable_ner: bool
    ner_type: str
    enable_keywords: bool
    retrieval_mode: str
    enable_bm25: bool
    rrf_k: int
    enable_prf: bool
    prf_docs: int
    prf_terms: int
    enable_rrf_signal_rerank: bool
    reranker_backend: str
    reranker_model: str
    rerank_candidates: int
    answer_eval: bool
    answer_k: int
    answer_model: Optional[str]
    judge_model: Optional[str]
    output_path: Optional[Path]
    dummy_embeddings: bool


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate mnemotree retrieval and optional answer-level quality.",
    )
    parser.add_argument(
        "--data-dir",
        default="benchmarks/data",
        help="Directory containing memories.jsonl and test_queries.jsonl.",
    )
    parser.add_argument(
        "--memories-file",
        default="memories.jsonl",
        help="JSONL file with memory entries.",
    )
    parser.add_argument(
        "--queries-file",
        default="test_queries.jsonl",
        help="JSONL file with evaluation queries.",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated list of k values for @k metrics.",
    )
    parser.add_argument(
        "--store",
        default="chroma",
        choices=["chroma", "chroma-graph", "baseline-chroma", "neo4j", "sqlite-vec", "inmemory"],
        help="Backend store for evaluation.",
    )
    parser.add_argument(
        "--mode",
        default="lite",
        choices=["lite", "pro"],
        help="MemoryCore mode (lite uses local embeddings; pro uses LLM embeddings).",
    )
    parser.add_argument(
        "--no-scoring",
        action="store_true",
        help="Disable MemoryScoring reranking/filtering.",
    )
    ner_group = parser.add_mutually_exclusive_group()
    ner_group.add_argument(
        "--enable-ner",
        action="store_true",
        default=False,
        dest="enable_ner",
        help="Enable NER-based entity retrieval (requires spaCy model).",
    )
    ner_group.add_argument(
        "--disable-ner",
        action="store_false",
        dest="enable_ner",
        help="Disable NER-based entity retrieval.",
    )
    parser.add_argument(
        "--ner-type",
        choices=["spacy", "llm", "gliner", "flair", "distilbert"],
        default="spacy",
        help="NER implementation type: spacy (fast, generic), llm (slow, domain-aware), gliner (fast, customizable), flair (torch-native), or distilbert (balanced speed+quality).",
    )

    keywords_group = parser.add_mutually_exclusive_group()
    keywords_group.add_argument(
        "--enable-keywords",
        action="store_true",
        default=False,
        dest="enable_keywords",
        help="Enable keyword extraction (requires spaCy model).",
    )
    keywords_group.add_argument(
        "--disable-keywords",
        action="store_false",
        dest="enable_keywords",
        help="Disable keyword extraction.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["basic", "hybrid", "rrf"],
        default="basic",
        help="Retrieval strategy: basic (vector+entity) or hybrid (dense+entity+BM25 RRF fusion). Use rrf as a legacy alias for hybrid.",
    )
    parser.add_argument(
        "--enable-bm25",
        action="store_true",
        help="Enable BM25 sparse retrieval (only used with --retrieval-mode hybrid).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k constant (typical default: 60).",
    )
    parser.add_argument(
        "--enable-prf",
        action="store_true",
        help="Enable pseudo-relevance feedback (PRF) query expansion for BM25 (only used with --retrieval-mode hybrid and --enable-bm25).",
    )
    parser.add_argument(
        "--prf-docs",
        type=int,
        default=5,
        help="Number of top BM25 documents to use for PRF term extraction.",
    )
    parser.add_argument(
        "--prf-terms",
        type=int,
        default=8,
        help="Number of PRF expansion terms to add to the BM25 query.",
    )
    parser.add_argument(
        "--rrf-rerank-signals",
        action="store_true",
        help="After hybrid fusion, rerank by dense similarity with small keyword/entity/RRF boosts.",
    )
    parser.add_argument(
        "--reranker-backend",
        choices=["none", "flashrank"],
        default="none",
        help="Optional reranker backend applied after fusion in RRF mode.",
    )
    parser.add_argument(
        "--reranker-model",
        default="ms-marco-TinyBERT-L-2-v2",
        help="Reranker model name (FlashRank).",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=50,
        help="Number of top fused candidates to rerank.",
    )
    parser.add_argument(
        "--answer-eval",
        action="store_true",
        help="Generate answers and score faithfulness/relevance with an LLM judge.",
    )
    parser.add_argument(
        "--answer-k",
        type=int,
        default=5,
        help="Number of retrieved contexts to pass into answer evaluation.",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="LLM model name used to generate answers.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini"),
        help="LLM model name used to judge answers.",
    )
    parser.add_argument(
        "--dummy-embeddings",
        action="store_true",
        help="Use a deterministic local embedding function (for tests/CI).",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/evaluation.json",
        help="Output JSON path for evaluation results.",
    )
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    if not k_values:
        k_values = DEFAULT_K_VALUES
    k_values = sorted(set(k_values))

    output_path = Path(args.output) if args.output else None

    return EvalConfig(
        data_dir=Path(args.data_dir),
        memories_file=args.memories_file,
        queries_file=args.queries_file,
        k_values=k_values,
        store_type=args.store,
        mode=args.mode,
        scoring=not args.no_scoring,
        enable_ner=args.enable_ner,
        ner_type=args.ner_type,
        enable_keywords=args.enable_keywords,
        retrieval_mode=args.retrieval_mode,
        enable_bm25=args.enable_bm25,
        rrf_k=args.rrf_k,
        enable_prf=args.enable_prf,
        prf_docs=args.prf_docs,
        prf_terms=args.prf_terms,
        enable_rrf_signal_rerank=args.rrf_rerank_signals,
        reranker_backend=args.reranker_backend,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        answer_eval=args.answer_eval,
        answer_k=args.answer_k,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        output_path=output_path,
        dummy_embeddings=bool(args.dummy_embeddings),
    )


class DummyEmbeddings:
    """Deterministic hashed bag-of-words embeddings (fast, no external deps)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    async def aembed_query(self, text: str) -> List[float]:
        tokens = [t.lower() for t in text.split() if t.strip()]
        vec = [0.0] * self.dim
        for token in tokens:
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class InMemoryVectorStore:
    """Simple in-memory vector store for benchmark smoke tests."""

    def __init__(self):
        self._memories: Dict[str, Any] = {}

    async def initialize(self) -> None:
        return None

    async def store_memory(self, memory):
        self._memories[memory.memory_id] = memory

    async def get_memory(self, memory_id: str):
        return self._memories.get(memory_id)

    async def delete_memory(self, memory_id: str, *, _cascade: bool = False) -> bool:
        return self._memories.pop(memory_id, None) is not None

    async def update_connections(self, memory_id: str, **kwargs):
        return None

    async def update_memory_metadata(self, memory_id: str, metadata: Dict[str, Any]):
        memory = self._memories.get(memory_id)
        if not memory:
            return False
        for key, value in metadata.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        return True

    async def query_by_entities(self, _entities, _limit: int = 10):
        return []

    async def query_memories(self, _query):
        return []

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        _filters: Optional[Dict[str, Any]] = None,
    ):
        def cosine(a: Sequence[float], b: Sequence[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for memory in self._memories.values():
            emb = getattr(memory, "embedding", None) or []
            scored.append((cosine(query_embedding, emb), memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:top_k]]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_memories(path: Path) -> List[MemoryRecord]:
    records: List[MemoryRecord] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(
                MemoryRecord(
                    content=data["content"],
                    memory_type=data.get("memory_type"),
                    concepts=list(data.get("concepts") or data.get("tags") or []),
                    entities=data.get("entities") or {},
                    emotion=data.get("emotion"),
                    importance=_safe_float(data.get("importance")),
                )
            )
    return records


def load_queries(path: Path) -> List[QueryCase]:
    queries: List[QueryCase] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(
                QueryCase(
                    query=data["query"],
                    expected_memories=list(data.get("expected_memories") or []),
                    expected_ids=list(data.get("expected_ids") or []),
                    expected_answer=data.get("expected_answer"),
                    description=data.get("description", ""),
                    query_type=data.get("query_type"),
                )
            )
    return queries


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    denom = min(k, len(retrieved)) if retrieved else k
    if denom == 0:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / denom


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def mrr_score(retrieved: Sequence[str], relevant: set[str]) -> float:
    for idx, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rels = [1.0 if item in relevant else 0.0 for item in retrieved[:k]]
    if not rels:
        return 0.0
    dcg = sum(rel / (math.log2(idx + 1)) for idx, rel in enumerate(rels, 1))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / (math.log2(idx + 1)) for idx, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def _stringify_k_metrics(metrics: Dict[int, float]) -> Dict[str, float]:
    return {str(k): value for k, value in metrics.items()}


def _truncate(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class AnswerJudgmentModel:
    def __init__(self, judge_model: str):
        from pydantic import BaseModel, Field
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        class AnswerJudgment(BaseModel):
            answer_relevance: float = Field(
                description="Score between 0 and 1 for how well the answer addresses the question."
            )
            faithfulness: float = Field(
                description="Score between 0 and 1 for how well the answer is supported by the contexts."
            )
            explanation: str = Field(description="Short explanation for the scores.")
            unsupported_claims: List[str] = Field(
                description="Any claims not supported by the contexts."
            )

        self._model = AnswerJudgment
        parser = JsonOutputParser(pydantic_object=AnswerJudgment)
        prompt = PromptTemplate(
            template=(
                "You are evaluating a QA system.\n\n"
                "Question:\n{question}\n\n"
                "Answer:\n{answer}\n\n"
                "Contexts:\n{contexts}\n\n"
                "Score the answer for:\n"
                "- answer_relevance: how well the answer addresses the question\n"
                "- faithfulness: how well the answer is supported by the contexts\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["question", "answer", "contexts"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.chain = prompt | ChatOpenAI(model=judge_model, temperature=0) | parser

    async def judge(self, question: str, answer: str, contexts: str) -> Dict[str, Any]:
        result = await self.chain.ainvoke(
            {"question": question, "answer": answer, "contexts": contexts}
        )
        return self._model(**result).model_dump()


class AnswerGenerator:
    def __init__(self, model_name: str):
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = PromptTemplate(
            template=(
                "Answer the question using only the contexts below.\n"
                "If the answer is not present, say \"I don't know.\"\n\n"
                "Question:\n{question}\n\n"
                "Contexts:\n{contexts}\n\n"
                "Answer:"
            ),
            input_variables=["question", "contexts"],
        )
        self.chain = prompt | ChatOpenAI(model=model_name, temperature=0)

    async def generate(self, question: str, contexts: str) -> str:
        response = await self.chain.ainvoke({"question": question, "contexts": contexts})
        return getattr(response, "content", str(response))


async def build_store(store_type: str, data_dir: Path):
    if store_type == "chroma":
        from mnemotree.store.chromadb_store import ChromaMemoryStore

        store = ChromaMemoryStore()
        await store.initialize()
        return store
    if store_type == "chroma-graph":
        from mnemotree.store.chromadb_store import ChromaMemoryStore

        persist_dir = data_dir / "chroma_graph_benchmark"
        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)

        store = ChromaMemoryStore(
            persist_directory=str(persist_dir),
            enable_graph_index=True,
        )
        await store.initialize()
        return store
    if store_type == "baseline-chroma":
        from mnemotree.store.baseline.chromadb_store import BaselineChromaStore

        store = BaselineChromaStore()
        await store.initialize()
        return store
    if store_type == "neo4j":
        from mnemotree.store.neo4j_store import Neo4jMemoryStore

        uri = os.getenv("MNEMOTREE_NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("MNEMOTREE_NEO4J_USER", "neo4j")
        password = os.getenv("MNEMOTREE_NEO4J_PASSWORD", "password")
        store = Neo4jMemoryStore(uri=uri, user=user, password=password)
        await store.initialize()
        return store
    if store_type == "sqlite-vec":
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

        db_path = data_dir / "sqlite_vec_benchmark.sqlite"
        if db_path.exists():
            db_path.unlink()
        store = SQLiteVecMemoryStore(db_path=db_path)
        await store.initialize()
        return store
    if store_type == "inmemory":
        store = InMemoryVectorStore()
        await store.initialize()
        return store
    raise ValueError(f"Unsupported store type: {store_type}")


async def seed_memories(
    memory_core,
    memories: List[MemoryRecord],
) -> int:
    from mnemotree.core.models import MemoryType

    count = 0
    for record in memories:
        memory_type = None
        if record.memory_type:
            try:
                memory_type = MemoryType(record.memory_type)
            except ValueError:
                memory_type = None
        context = {}
        if record.entities:
            context["entities"] = record.entities
        if record.emotion:
            context["emotion"] = record.emotion
        await memory_core.remember(
            content=record.content,
            memory_type=memory_type,
            tags=record.concepts or None,
            importance=record.importance,
            context=context or None,
            analyze=False,
            summarize=False,
        )
        count += 1
    return count


def build_context_block(memories: Sequence[Any]) -> str:
    if not memories:
        return "None."
    lines = []
    for idx, memory in enumerate(memories, 1):
        lines.append(f"[{idx}] {memory.content}")
    return "\n".join(lines)


def _relevant_set_for_query(query: QueryCase) -> Tuple[set[str], str]:
    if query.expected_ids:
        return set(query.expected_ids), "memory_id"
    return set(query.expected_memories), "content"


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)

    sorted_values = sorted(values)
    # Linear interpolation between closest ranks.
    pos = (len(sorted_values) - 1) * (percentile / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return (sorted_values[lo] * (1.0 - weight)) + (sorted_values[hi] * weight)


async def evaluate_queries(
    memory_core,
    queries: List[QueryCase],
    k_values: List[int],
    scoring: bool,
    answer_eval: bool,
    answer_k: int,
    answer_model: Optional[str],
    judge_model: Optional[str],
) -> Dict[str, Any]:
    max_k = max(k_values)
    answer_k = min(answer_k, max_k) if answer_k > 0 else max_k
    answer_generator = None
    answer_judge = None
    if answer_eval:
        if not answer_model or not judge_model:
            raise ValueError("answer-model and judge-model must be set for answer eval")
        answer_generator = AnswerGenerator(answer_model)
        answer_judge = AnswerJudgmentModel(judge_model)

    per_query_results = []

    for query in queries:
        recall_start = time.perf_counter()
        retrieved = await memory_core.recall(
            query.query,
            limit=max_k,
            scoring=scoring,
            update_access=False,
        )
        recall_ms = (time.perf_counter() - recall_start) * 1000.0
        query_embedding = await memory_core.get_embedding(query.query)

        relevant_set, relevance_key = _relevant_set_for_query(query)
        retrieved_keys = [
            getattr(item, relevance_key) if relevance_key == "memory_id" else item.content
            for item in retrieved
        ]

        metrics_precision = {
            k: precision_at_k(retrieved_keys, relevant_set, k) for k in k_values
        }
        metrics_recall = {
            k: recall_at_k(retrieved_keys, relevant_set, k) for k in k_values
        }
        metrics_ndcg = {
            k: ndcg_at_k(retrieved_keys, relevant_set, k) for k in k_values
        }

        semantic_similarities = [
            cosine_similarity(query_embedding, item.embedding or [])
            for item in retrieved
        ]
        semantic_similarity = (
            sum(semantic_similarities) / len(semantic_similarities)
            if semantic_similarities
            else 0.0
        )

        result = {
            "query": query.query,
            "description": query.description,
            "query_type": query.query_type,
            "timing": {
                "recall_ms": recall_ms,
            },
            "metrics": {
                METRIC_PRECISION: _stringify_k_metrics(metrics_precision),
                METRIC_RECALL: _stringify_k_metrics(metrics_recall),
                METRIC_NDCG: _stringify_k_metrics(metrics_ndcg),
                "mrr": mrr_score(retrieved_keys, relevant_set),
                "semantic_similarity": semantic_similarity,
            },
            "retrieved": [
                {
                    "memory_id": item.memory_id,
                    "content": _truncate(item.content),
                    **(
                        {
                            "connection_depth": (item.context or {}).get(
                                "connection_depth"
                            ),
                            "matching_entities": (item.context or {}).get(
                                "matching_entities"
                            ),
                        }
                        if isinstance(getattr(item, "context", None), dict)
                        and (
                            "connection_depth" in (item.context or {})
                            or "matching_entities" in (item.context or {})
                        )
                        else {}
                    ),
                }
                for item in retrieved
            ],
        }

        if answer_eval and answer_generator and answer_judge:
            contexts = build_context_block(retrieved[:answer_k])
            answer = await answer_generator.generate(query.query, contexts)
            judgment = await answer_judge.judge(query.query, answer, contexts)
            context_precision = precision_at_k(retrieved_keys, relevant_set, answer_k)
            context_recall = recall_at_k(retrieved_keys, relevant_set, answer_k)
            result["answer_eval"] = {
                "answer": answer,
                "answer_relevance": judgment["answer_relevance"],
                "faithfulness": judgment["faithfulness"],
                "context_precision": context_precision,
                "context_recall": context_recall,
                "judge_explanation": judgment["explanation"],
                "unsupported_claims": judgment["unsupported_claims"],
            }

        per_query_results.append(result)

    return {
        "per_query_results": per_query_results,
    }


def aggregate_results(
    per_query_results: List[Dict[str, Any]],
    k_values: List[int],
    answer_eval: bool,
) -> Dict[str, Any]:
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    precision_summary = {}
    recall_summary = {}
    ndcg_summary = {}

    for k in k_values:
        precision_summary[k] = mean(
            [r["metrics"]["precision@k"][str(k)] for r in per_query_results]
        )
        recall_summary[k] = mean(
            [r["metrics"]["recall@k"][str(k)] for r in per_query_results]
        )
        ndcg_summary[k] = mean(
            [r["metrics"]["ndcg@k"][str(k)] for r in per_query_results]
        )

    summary = {
        METRIC_PRECISION: _stringify_k_metrics(precision_summary),
        METRIC_RECALL: _stringify_k_metrics(recall_summary),
        METRIC_NDCG: _stringify_k_metrics(ndcg_summary),
        "mrr": mean([r["metrics"]["mrr"] for r in per_query_results]),
        "semantic_similarity": mean(
            [r["metrics"]["semantic_similarity"] for r in per_query_results]
        ),
    }

    recall_latencies_ms = [
        float(r.get("timing", {}).get("recall_ms", 0.0)) for r in per_query_results
    ]
    summary["latency_ms"] = {
        "recall": {
            "avg": mean(recall_latencies_ms),
            "p50": _percentile(recall_latencies_ms, 50),
            "p95": _percentile(recall_latencies_ms, 95),
            "min": min(recall_latencies_ms) if recall_latencies_ms else 0.0,
            "max": max(recall_latencies_ms) if recall_latencies_ms else 0.0,
        }
    }

    if answer_eval:
        summary["answer_eval"] = {
            "answer_relevance": mean(
                [r["answer_eval"]["answer_relevance"] for r in per_query_results]
            ),
            "faithfulness": mean(
                [r["answer_eval"]["faithfulness"] for r in per_query_results]
            ),
            "context_precision": mean(
                [r["answer_eval"]["context_precision"] for r in per_query_results]
            ),
            "context_recall": mean(
                [r["answer_eval"]["context_recall"] for r in per_query_results]
            ),
        }

    return summary


async def run_benchmark(config: EvalConfig) -> Dict[str, Any]:
    from mnemotree.core.memory import MemoryCore, ModeDefaultsConfig, NerConfig, RetrievalConfig

    retrieval_mode = "hybrid" if config.retrieval_mode == "rrf" else config.retrieval_mode

    memories_path = config.data_dir / config.memories_file
    queries_path = config.data_dir / config.queries_file

    memories = load_memories(memories_path)
    queries = load_queries(queries_path)

    store = await build_store(config.store_type, config.data_dir)
    
    # Setup NER if enabled
    ner = None
    if config.enable_ner:
        if config.ner_type == "llm":
            from langchain_openai import ChatOpenAI
            from mnemotree.ner.llm import LangchainLLMNER
            llm = ChatOpenAI(model=config.answer_model or DEFAULT_MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
            ner = LangchainLLMNER(llm)
        elif config.ner_type == "gliner":
            from mnemotree.ner.gliner import GLiNERNER
            # Use culinary-specific entity types to match the test data
            entity_types = ["person", "location", "dish", "ingredient", "cuisine", "food", "spice"]
            ner = GLiNERNER(entity_types=entity_types, threshold=0.3)
        elif config.ner_type == "flair":
            from mnemotree.ner.flair import FlairNER
            # Use XLM-RoBERTa large fine-tuned on OntoNotes (similar to Flair's approach)
            ner = FlairNER(model_name="xlm-roberta-large-finetuned-conll03-english")
        elif config.ner_type == "distilbert":
            from mnemotree.ner.distilbert import DistilBERTNER
            ner = DistilBERTNER(model_name="dslim/distilbert-NER")
        # else: spacy is the default, let MemoryCore create it
    
    embeddings = DummyEmbeddings() if config.dummy_embeddings else None

    memory_core = MemoryCore(
        store=store,
        embeddings=embeddings,
        mode_defaults=ModeDefaultsConfig(
            mode=config.mode,
            enable_keywords=config.enable_keywords,
        ),
        ner_config=NerConfig(
            ner=ner,
            enable_ner=config.enable_ner,
        ),
        retrieval_config=RetrievalConfig(
            retrieval_mode=retrieval_mode,
            enable_bm25=config.enable_bm25,
            rrf_k=config.rrf_k,
            enable_prf=config.enable_prf,
            prf_docs=config.prf_docs,
            prf_terms=config.prf_terms,
            enable_rrf_signal_rerank=config.enable_rrf_signal_rerank,
            reranker_backend=config.reranker_backend,
            reranker_model=config.reranker_model,
            rerank_candidates=config.rerank_candidates,
        ),
    )

    num_memories = await seed_memories(memory_core, memories)

    evaluation = await evaluate_queries(
        memory_core=memory_core,
        queries=queries,
        k_values=config.k_values,
        scoring=config.scoring,
        answer_eval=config.answer_eval,
        answer_k=config.answer_k,
        answer_model=config.answer_model,
        judge_model=config.judge_model,
    )

    summary = aggregate_results(
        evaluation["per_query_results"],
        config.k_values,
        config.answer_eval,
    )

    return {
        "summary": {
            "num_memories": num_memories,
            "num_queries": len(queries),
            "metrics": summary,
        },
        "per_query_results": evaluation["per_query_results"],
        "config": {
            "data_dir": str(config.data_dir),
            "memories_file": config.memories_file,
            "queries_file": config.queries_file,
            "k_values": config.k_values,
            "store": config.store_type,
            "mode": config.mode,
            "scoring": config.scoring,
            "enable_ner": config.enable_ner,
            "ner_type": config.ner_type,
            "enable_keywords": config.enable_keywords,
            "retrieval_mode": config.retrieval_mode,
            "enable_bm25": config.enable_bm25,
            "rrf_k": config.rrf_k,
            "enable_prf": config.enable_prf,
            "prf_docs": config.prf_docs,
            "prf_terms": config.prf_terms,
            "enable_rrf_signal_rerank": config.enable_rrf_signal_rerank,
            "reranker_backend": config.reranker_backend,
            "reranker_model": config.reranker_model,
            "rerank_candidates": config.rerank_candidates,
            "answer_eval": config.answer_eval,
            "answer_k": config.answer_k,
            "answer_model": config.answer_model,
            "judge_model": config.judge_model,
            "dummy_embeddings": config.dummy_embeddings,
        },
    }


def main() -> None:
    config = parse_args()
    results = asyncio.run(run_benchmark(config))
    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with config.output_path.open("w") as handle:
            json.dump(results, handle, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
