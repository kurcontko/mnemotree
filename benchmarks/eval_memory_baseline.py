from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

# Add project root to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.models import MemoryItem
from src.core.memory import MemoryCore
from src.store.neo4j_store import Neo4jMemoryStore
from src.store.baseline.neo4j_store import BaselineNeo4jMemoryStore
from src.core.scoring import MemoryScoring


@dataclass
class TestQuery:
    query: str
    expected_memories: List[str]
    description: str


class RelevanceEvaluation(BaseModel):
    relevance_score: float = Field(
        description="Float between 0-1 indicating how relevant the memory is to the query"
    )
    explanation: str = Field(
        description="Brief explanation of the relevance score"
    )
    missing_aspects: List[str] = Field(
        description="List of critical elements that are missing from the memory"
    )


class RetrievalMetrics(BaseModel):
    precision: float = Field(description="Precision of retrieved results")
    recall: float = Field(description="Recall of relevant memories")
    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain")
    semantic_similarity: float = Field(description="Average semantic similarity")
    llm_relevance: float = Field(description="LLM-based relevance score")


@dataclass
class EvaluationResult:
    query: TestQuery
    retrieved_memories: List[MemoryItem]
    metrics: RetrievalMetrics
    analysis: Dict[str, Any]


class MemoryEvaluator:
    """Evaluates memory system retrieval performance using JSONL test datasets"""
    
    def __init__(
        self,
        memory_core: MemoryCore,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        data_dir: str
    ):
        self.memory_core = memory_core
        self.llm = llm
        self.embeddings = embeddings
        self.data_dir = Path(data_dir)
        self.relevance_chain = self._create_relevance_chain()
        
    def _create_relevance_chain(self) -> Runnable:
        """Create chain for evaluating memory relevance"""
        parser = JsonOutputParser(pydantic_object=RelevanceEvaluation)
        
        template = """Evaluate the relevance of retrieved memories to a query.
        Consider:
        1. Direct relevance - Does the memory content match the query intent?
        2. Completeness - Does it fully address the query?
        3. Accuracy - Is the information accurate and appropriate?
        
        Query: {query}
        
        Retrieved Memory:
        {memory}
        
        Rate the relevance of this memory.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "memory"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt | self.llm | parser
        
    async def load_and_store_memories(self) -> int:
        """Load memories from JSONL and store in memory system"""
        memories_file = self.data_dir / "memories.jsonl"
        
        stored_count = 0
        with open(memories_file) as f:
            for line in f:
                memory_data = json.loads(line)
                await self.memory_core.remember(
                    content=memory_data["content"],
                    analyze=False,
                    summarize=False
                )
                stored_count += 1
                
        return stored_count
        
    def load_test_queries(self) -> List[TestQuery]:
        """Load test queries from JSONL"""
        queries_file = self.data_dir / "test_queries.jsonl"
        
        queries = []
        with open(queries_file) as f:
            for line in f:
                data = json.loads(line)
                test_query = data.get("query", "")
                expected_memories = data.get("expected_memories", [])
                description = data.get("description", "")
                queries.append(TestQuery(
                    query=test_query,
                    expected_memories=expected_memories,
                    description=description
                ))
                
        return queries
        
    async def evaluate_query(
        self,
        test_query: TestQuery,
        limit: int = 10
    ) -> EvaluationResult:
        """Evaluate a single test query"""
        query_embedding = await self.embeddings.aembed_query(test_query.query)
        retrieved_memories = await self.memory_core.recall(
            test_query.query,
            limit=limit,
            scoring=False
        )
        
        metrics = await self._calculate_metrics(
            test_query,
            retrieved_memories,
            query_embedding
        )
        
        analysis = self.analyze_results(
            test_query,
            retrieved_memories,
            query_embedding
        )
        
        return EvaluationResult(
            query=test_query,
            retrieved_memories=retrieved_memories,
            metrics=metrics,
            analysis=analysis
        )
        
    async def _calculate_metrics(
        self,
        query: TestQuery,
        retrieved: List[MemoryItem],
        query_embedding: List[float]
    ) -> RetrievalMetrics:
        """Calculate comprehensive retrieval metrics"""
        retrieved_contents = [m.content for m in retrieved]
        relevant_retrieved = set(retrieved_contents) & set(query.expected_memories)
        
        precision = len(relevant_retrieved) / len(retrieved) if retrieved else 0
        recall = len(relevant_retrieved) / len(query.expected_memories) if query.expected_memories else 0
        
        # Calculate MRR
        mrr = 0.0
        for i, memory in enumerate(retrieved, 1):
            if memory.content in query.expected_memories:
                mrr = 1.0 / i
                break
                
        # Calculate NDCG
        relevance_vector = [
            1.0 if m.content in query.expected_memories else 0.0
            for m in retrieved
        ]
        ndcg = self._calculate_ndcg(relevance_vector)
        
        # Calculate semantic similarity
        semantic_similarities = [
            self._cosine_similarity(query_embedding, m.embedding)
            for m in retrieved
            if m.embedding is not None
        ]
        avg_semantic_similarity = np.mean(semantic_similarities) if semantic_similarities else 0.0
        
        # Get LLM-based relevance scores for top 3 results
        relevance_scores = []
        for memory in retrieved[:3]:
            try:
                result = await self.relevance_chain.ainvoke({
                    "query": query.query,
                    "memory": memory.content
                })
                # Parse the result into RelevanceEvaluation model
                evaluation = RelevanceEvaluation(**result)
                relevance_scores.append(evaluation.relevance_score)
            except Exception as e:
                print(f"Error evaluating relevance: {e}")
                
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            mrr=mrr,
            ndcg=ndcg,
            semantic_similarity=avg_semantic_similarity,
            llm_relevance=avg_relevance
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
            
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        
        return float(dot_product / (norm1 * norm2)) if norm1 and norm2 else 0.0
        
    def _calculate_ndcg(self, relevance_scores: List[float], k: int = None) -> float:
        """Calculate NDCG@k metric"""
        if not relevance_scores:
            return 0.0
            
        if k is not None:
            relevance_scores = relevance_scores[:k]
            
        dcg = sum(
            (2 ** rel - 1) / np.log2(i + 1)
            for i, rel in enumerate(relevance_scores, 1)
        )
        
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 1)
            for i, rel in enumerate(ideal_scores, 1)
        )
            
        return dcg / idcg if idcg > 0 else 0.0

    def analyze_results(
        self,
        query: TestQuery,
        retrieved: List[MemoryItem],
        query_embedding: List[float]
    ) -> Dict[str, Any]:
        """Analyze retrieval results"""
        retrieved_contents = [m.content for m in retrieved]
        
        # Calculate semantic similarity distribution
        similarities = [
            self._cosine_similarity(query_embedding, m.embedding)
            for m in retrieved
            if m.embedding is not None
        ]
        
        return {
            "num_retrieved": len(retrieved),
            "num_expected": len(query.expected_memories),
            "retrieved_memory_types": self._count_memory_types(retrieved),
            "missing_memories": list(
                set(query.expected_memories) - set(retrieved_contents)
            ),
            "importance_stats": self._analyze_importance(retrieved),
            "similarity_stats": {
                "mean": float(np.mean(similarities)) if similarities else 0.0,
                "std": float(np.std(similarities)) if similarities else 0.0,
                "min": float(np.min(similarities)) if similarities else 0.0,
                "max": float(np.max(similarities)) if similarities else 0.0
            }
        }

    def _count_memory_types(self, memories: List[MemoryItem]) -> Dict[str, int]:
        """Count occurrences of each memory type"""
        type_counts = {}
        for memory in memories:
            type_str = memory.memory_type.value
            type_counts[type_str] = type_counts.get(type_str, 0) + 1
        return type_counts
        
    def _analyze_importance(self, memories: List[MemoryItem]) -> Dict[str, float]:
        """Analyze importance scores of retrieved memories"""
        if not memories:
            return {}
            
        scores = [m.importance for m in memories]
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores))
        }

async def run_evaluation(
    memory_core: MemoryCore,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    data_dir: str
) -> Dict[str, Any]:
    """Run complete evaluation process"""
    evaluator = MemoryEvaluator(memory_core, llm, embeddings, data_dir)
    
    print("Loading and storing memories...")
    num_stored = await evaluator.load_and_store_memories()
    
    print("Loading test queries...")
    test_queries = evaluator.load_test_queries()
    
    print("Evaluating queries...")
    results = []
    for query in test_queries:
        result = await evaluator.evaluate_query(query)
        results.append(result)
        print(f"Evaluated query: {query.query[:50]}...")
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([getattr(r.metrics, metric) for r in results])
        for metric in RetrievalMetrics.__fields__
    }
    
    return {
        "summary": {
            "num_memories": num_stored,
            "num_queries": len(test_queries),
            "avg_metrics": avg_metrics
        },
        "per_query_results": [
            {
                "query": r.query.query,
                "description": r.query.description,
                "metrics": r.metrics.dict(),
                "analysis": r.analysis
            }
            for r in results
        ]
    }

if __name__ == "__main__":
    async def main():
        from langchain_openai import ChatOpenAI
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize components
        llm_primary = ChatOpenAI(model="gpt-4o")
        llm_secondary = ChatOpenAI(model="gpt-4o-mini")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        store = BaselineNeo4jMemoryStore(
            uri="bolt://localhost:7688",
            user="neo4j",
            password="testpassword"
        )
        await store.initialize()
        
        # For benchmarking purposes, remove recency weight and increase importance weight
        memory_scoring = MemoryScoring(
            importance_weight=0.5,
            recency_weight=0,
            access_weight=0,
            emotion_weight=0.15,
            context_weight=0.15,
            novelty_weight=0.2,
            score_threshold=0.65,
        )
        memory_core = MemoryCore(
            llm=llm_secondary,
            embeddings=embeddings,
            store=store,
            memory_scoring=memory_scoring,
            ner=False
        )
        
        # Run evaluation
        results = await run_evaluation(
            memory_core=memory_core,
            llm=llm_primary,
            embeddings=embeddings,
            data_dir="test_data"
        )
        
        # Save results
        with open("evaluation_results_baseline_neo4j.json", "w") as f:
            json.dump(results, f, indent=2)
        
    asyncio.run(main())