"""
Example: Advanced memory system with all innovative features.

Demonstrates:
1. Hybrid retrieval with reranking
2. Memory consolidation (sleep cycles)
3. Conflict detection and truth maintenance
4. Adaptive decay with spaced repetition
5. Context-aware write gating
6. Memory evaluation metrics
"""

import asyncio
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mnemotree.core.adaptive_decay import (
    AdaptiveImportanceSystem,
    DecayParameters,
)
from mnemotree.core.consolidation import ConsolidationConfig, MemoryConsolidator
from mnemotree.core.evaluation import (
    MemoryEvaluator,
    SyntheticDatasetGenerator,
)
from mnemotree.core.hybrid_retrieval import FusionStrategy, HybridRetriever
from mnemotree.core.memory import MemoryCore
from mnemotree.core.models import MemoryItem, MemoryType
from mnemotree.core.truth_maintenance import ClaimsRegistry, ResolutionStrategy
from mnemotree.core.write_gate import (
    ContextAwareWriteGate,
    WriteDecision,
    WritePolicy,
)
from mnemotree.rerankers import CrossEncoderReranker
from mnemotree.store.chromadb_store import ChromaDBStore


async def main():
    """Run advanced memory system demonstration."""

    print("🌳 Mnemotree - Advanced Memory System Demo\n")

    # Initialize components
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    embeddings = OpenAIEmbeddings()
    store = ChromaDBStore(
        collection_name="advanced_demo",
        embeddings=embeddings,
    )

    # Initialize memory core
    memory = MemoryCore(store=store, llm=llm, embeddings=embeddings)

    # ========================================
    # 1. CONTEXT-AWARE WRITE GATING
    # ========================================
    print("=" * 60)
    print("1. Context-Aware Write Gating")
    print("=" * 60)

    write_gate = ContextAwareWriteGate(
        policy=WritePolicy.balanced()
    )

    # Try storing different quality memories
    test_memories = [
        {
            "content": "The Eiffel Tower is located in Paris, France and was completed in 1889.",
            "importance": 0.7,
            "confidence": 0.9,
        },
        {
            "content": "ok",  # Too short
            "importance": 0.5,
            "confidence": 0.8,
        },
        {
            "content": "The Eiffel Tower is in Paris.",  # Redundant with first
            "importance": 0.6,
            "confidence": 0.8,
        },
    ]

    stored_ids = []
    for i, mem_data in enumerate(test_memories):
        mem = MemoryItem(
            content=mem_data["content"],
            memory_type=MemoryType.SEMANTIC,
            importance=mem_data["importance"],
            confidence=mem_data["confidence"],
            embedding=await memory.embedder.aembed_query(mem_data["content"]),
        )

        # Evaluate with write gate
        existing = [m for m in await store.get_all_memories() if m.memory_id in stored_ids]
        gate_result = await write_gate.evaluate(mem, existing_memories=existing)

        print(f"\nMemory {i+1}: \"{mem_data['content'][:50]}...\"")
        print(f"  Decision: {gate_result.decision}")
        print(f"  Score: {gate_result.score:.2f}")
        print(f"  Reasons: {', '.join(gate_result.reasons)}")

        if gate_result.decision == WriteDecision.ACCEPT:
            stored_mem = await memory.remember(
                content=mem_data["content"],
                memory_type=MemoryType.SEMANTIC,
                importance=mem_data["importance"],
            )
            stored_ids.append(stored_mem.memory_id)
            print(f"  ✓ Stored with ID: {stored_mem.memory_id[:8]}")
        else:
            print("  ✗ Not stored")

    # ========================================
    # 2. ADAPTIVE IMPORTANCE & DECAY
    # ========================================
    print("\n" + "=" * 60)
    print("2. Adaptive Importance & Decay")
    print("=" * 60)

    adaptive_system = AdaptiveImportanceSystem(
        decay_params=DecayParameters.default(),
        enable_spaced_repetition=True,
    )

    # Add some memories with different patterns
    episodic_memories = []
    for i in range(5):
        mem = await memory.remember(
            content=f"I had coffee at the cafe on day {i+1}. The weather was nice.",
            memory_type=MemoryType.EPISODIC,
            importance=0.4 + (i * 0.1),
        )
        episodic_memories.append(mem)

        # Simulate different access patterns
        for _ in range(i):  # More accesses for later memories
            mem.update_access()

    print("\nMemory decay analysis:")
    for mem in episodic_memories:
        decay_rate = adaptive_system.calculate_adaptive_decay(mem)
        novelty = adaptive_system.assess_novelty(mem)

        print(f"\nMemory: {mem.content[:40]}...")
        print(f"  Importance: {mem.importance:.2f}")
        print(f"  Access count: {mem.access_count}")
        print(f"  Decay rate: {decay_rate:.3f}")
        print(f"  Novelty: {novelty.value}")

        # Register high-value memories for spaced repetition
        if mem.importance > 0.6:
            adaptive_system.register_for_spaced_repetition(mem)
            print("  ✓ Registered for spaced repetition")

    print("\nSpaced repetition stats:")
    stats = adaptive_system.get_decay_statistics()
    print(f"  Memories tracked: {stats['memories_in_sr']}")
    print(f"  Due for review: {stats['due_reviews']}")

    # ========================================
    # 3. HYBRID RETRIEVAL WITH RERANKING
    # ========================================
    print("\n" + "=" * 60)
    print("3. Hybrid Retrieval with Reranking")
    print("=" * 60)

    # Initialize hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_weight=0.5,
        entity_weight=0.3,
        graph_weight=0.2,
        fusion_strategy=FusionStrategy.RRF,
        reranker=CrossEncoderReranker(),
    )

    query = "Tell me about coffee and cafes"

    # Get candidates from different sources
    vector_results = await memory.recall(query, limit=10)
    vector_candidates = [(m, 1.0 - i*0.1) for i, m in enumerate(vector_results)]

    # For demo, use same results as entity candidates
    entity_candidates = [(m, 0.8 - i*0.05) for i, m in enumerate(vector_results[:5])]

    print(f"\nQuery: \"{query}\"")
    print(f"Vector candidates: {len(vector_candidates)}")
    print(f"Entity candidates: {len(entity_candidates)}")

    # Run hybrid retrieval
    results = await hybrid_retriever.retrieve(
        query=query,
        vector_candidates=vector_candidates,
        entity_candidates=entity_candidates,
        top_k=5,
        apply_reranking=True,
    )

    print(f"\nTop {len(results)} results after hybrid retrieval:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.final_score:.3f}")
        print(f"   Content: {result.memory.content[:60]}...")
        print(f"   Stages: {', '.join([s.value for s in result.retrieval_stages])}")
        print(f"   Component scores: {result.scores}")

    # ========================================
    # 4. TRUTH MAINTENANCE & CONFLICT DETECTION
    # ========================================
    print("\n" + "=" * 60)
    print("4. Truth Maintenance & Conflict Detection")
    print("=" * 60)

    claims_registry = ClaimsRegistry(
        llm=llm,
        staleness_threshold_days=90,
    )

    # Add some memories with potential conflicts
    conflicting_memories = [
        await memory.remember(
            content="The cafe opens at 8 AM every morning.",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7,
        ),
        await memory.remember(
            content="The cafe starts serving at 9 AM on weekdays.",
            memory_type=MemoryType.SEMANTIC,
            importance=0.6,
        ),
    ]

    # Extract and register claims
    for mem in conflicting_memories:
        claims = await claims_registry.extract_claims(mem)
        print(f"\nExtracted {len(claims)} claims from: {mem.content[:50]}...")

        for claim in claims:
            print(f"  - {claim.statement[:60]}...")

        await claims_registry.register_claims(claims)

    # Check for conflicts
    conflicts = claims_registry.get_active_conflicts()
    print(f"\n{'Conflicts detected:' if conflicts else 'No conflicts detected'}")

    for conflict in conflicts:
        print(f"\nConflict ID: {conflict.conflict_id[:8]}")
        print(f"  Severity: {conflict.severity.value}")
        print(f"  Description: {conflict.description}")
        print(f"  Involved claims: {len(conflict.claim_ids)}")

        # Resolve conflict
        winner_id = await claims_registry.resolve_conflict(
            conflict.conflict_id,
            strategy=ResolutionStrategy.ENSEMBLE,
        )

        if winner_id:
            print(f"  ✓ Resolved - Winner: {winner_id[:8]}")

    # Registry statistics
    stats = claims_registry.get_statistics()
    print("\nClaims Registry Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # ========================================
    # 5. MEMORY CONSOLIDATION (SLEEP CYCLE)
    # ========================================
    print("\n" + "=" * 60)
    print("5. Memory Consolidation (Sleep Cycle)")
    print("=" * 60)

    consolidator = MemoryConsolidator(
        llm=llm,
        config=ConsolidationConfig(
            min_cluster_size=2,
            similarity_threshold=0.7,
            importance_threshold=0.3,
        ),
    )

    # Get episodic memories for consolidation
    all_memories = await store.get_all_memories()

    print(f"\nRunning consolidation on {len(all_memories)} memories...")

    consolidation_result = await consolidator.consolidate(all_memories)

    print("\nConsolidation Results:")
    print(f"  Total processed: {consolidation_result.total_memories_processed}")
    print(f"  Clusters formed: {consolidation_result.clusters_formed}")
    print(f"  Semantic memories created: {consolidation_result.semantic_memories_created}")
    print(f"  Memories deprecated: {consolidation_result.memories_deprecated}")
    print(f"  Duration: {consolidation_result.duration_seconds:.2f}s")

    if consolidation_result.cluster_summaries:
        print("\nCluster Summaries:")
        for i, summary in enumerate(consolidation_result.cluster_summaries[:3], 1):
            print(f"\n  Cluster {i}:")
            print(f"    Size: {summary['cluster_size']}")
            print(f"    Summary: {summary['summary'][:100]}...")
            print(f"    Tags: {', '.join(summary['common_tags'][:5])}")

    # ========================================
    # 6. MEMORY EVALUATION METRICS
    # ========================================
    print("\n" + "=" * 60)
    print("6. Memory Evaluation Metrics")
    print("=" * 60)

    evaluator = MemoryEvaluator()

    # Generate synthetic evaluation dataset
    print("\nGenerating synthetic evaluation queries...")
    dataset_generator = SyntheticDatasetGenerator(seed=42)

    all_memories = await store.get_all_memories()
    if len(all_memories) >= 10:
        eval_queries = dataset_generator.generate_queries(
            memories=all_memories,
            num_queries=10,
            relevance_threshold=0.7,
        )

        print(f"Generated {len(eval_queries)} evaluation queries")

        # Save queries for future use
        queries_path = Path("benchmarks/eval_queries.json")
        dataset_generator.save_queries(eval_queries, queries_path)
        print(f"Saved queries to {queries_path}")

        # Run benchmark
        print("\nRunning benchmark evaluation...")

        def retrieval_fn(query_text):
            # Sync wrapper for async recall
            return asyncio.run(memory.recall(query_text, limit=10))

        benchmark = evaluator.evaluate_benchmark(
            queries=eval_queries,
            retrieval_function=retrieval_fn,
            all_memories=all_memories,
            k_values=[1, 3, 5, 10],
        )

        print("\nBenchmark Results:")
        print(f"  Queries evaluated: {benchmark.total_queries}")
        print(f"  Total memories: {benchmark.total_memories}")

        print("\n  Retrieval Metrics:")
        for k, recall in benchmark.avg_recall_at_k.items():
            precision = benchmark.avg_precision_at_k[k]
            f1 = benchmark.avg_f1_at_k[k]
            print(f"    Recall@{k}: {recall:.3f}")
            print(f"    Precision@{k}: {precision:.3f}")
            print(f"    F1@{k}: {f1:.3f}")

        print(f"\n  MRR: {benchmark.avg_mrr:.3f}")
        print(f"  NDCG: {benchmark.avg_ndcg:.3f}")

        print("\n  System Health:")
        print(f"    Drift rate: {benchmark.drift_rate:.3f}")
        print(f"    Staleness ratio: {benchmark.staleness_ratio:.3f}")
        print(f"    Diversity score: {benchmark.diversity_score:.3f}")

        # Save benchmark
        benchmark_path = Path("benchmarks/results/benchmark_advanced.json")
        evaluator.save_benchmark(benchmark, benchmark_path)
        print(f"\n  ✓ Saved benchmark to {benchmark_path}")
    else:
        print("Not enough memories for evaluation (need at least 10)")

    print("\n" + "=" * 60)
    print("Demo completed! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
