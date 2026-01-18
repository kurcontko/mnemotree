"""
Quick start guide for advanced memory features.

This script demonstrates the minimal setup needed for each feature.
"""

import asyncio

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mnemotree.core import (
    AdaptiveImportanceSystem,
    ClaimsRegistry,
    ContextAwareWriteGate,
    CrossEncoderReranker,
    HybridRetriever,
    MemoryConsolidator,
    MemoryCore,
    MemoryEvaluator,
    WritePolicy,
)
from mnemotree.store.chromadb_store import ChromaDBStore


async def quick_start():
    """Quick start examples for each advanced feature."""

    # Setup
    llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()
    store = ChromaDBStore(collection_name="quickstart", embeddings=embeddings)
    memory = MemoryCore(store=store, llm=llm, embeddings=embeddings)

    # ============================================================
    # 1. HYBRID RETRIEVAL - Better search with multi-stage pipeline
    # ============================================================

    retriever = HybridRetriever(
        reranker=CrossEncoderReranker(),  # Optional: adds precision
    )

    query = "user preferences"
    vector_results = await memory.recall(query, limit=10)
    vector_candidates = [(m, 1.0) for m in vector_results]
    entity_candidates = [(m, 0.8) for m in vector_results[:5]]

    _results = await retriever.retrieve(
        query=query,
        vector_candidates=vector_candidates,
        entity_candidates=entity_candidates,
        top_k=5,
    )
    # → Get better ranked results with fusion and reranking

    # ============================================================
    # 2. MEMORY CONSOLIDATION - Clean up and summarize
    # ============================================================

    consolidator = MemoryConsolidator(llm=llm)

    # Run nightly or on-demand
    all_memories = await store.get_all_memories()
    result = await consolidator.consolidate(all_memories)
    # → Creates semantic memories, removes redundancy

    print(f"Created {result.semantic_memories_created} summaries")
    print(f"Deprecated {result.memories_deprecated} low-value memories")

    # ============================================================
    # 3. TRUTH MAINTENANCE - Detect and resolve conflicts
    # ============================================================

    registry = ClaimsRegistry(llm=llm)

    # Extract claims from memories
    memory_item = await memory.remember("The office opens at 9 AM")
    claims = await registry.extract_claims(memory_item)
    await registry.register_claims(claims)

    # Check for conflicts
    conflicts = registry.get_active_conflicts()
    for conflict in conflicts:
        await registry.resolve_conflict(conflict.conflict_id)
    # → Maintains factual consistency

    # ============================================================
    # 4. ADAPTIVE DECAY - Smart importance management
    # ============================================================

    adaptive = AdaptiveImportanceSystem()

    # Check decay and novelty
    _decay_rate = adaptive.calculate_adaptive_decay(memory_item)
    _novelty = adaptive.assess_novelty(memory_item)

    # Register important memories for spaced repetition
    if memory_item.importance > 0.7:
        adaptive.register_for_spaced_repetition(memory_item)

    # Update importance over time
    _new_importance = adaptive.update_importance(memory_item)
    # → Prevents important memories from fading

    # ============================================================
    # 5. WRITE GATING - Filter out noise
    # ============================================================

    gate = ContextAwareWriteGate(
        policy=WritePolicy.balanced()  # or .strict() or .permissive()
    )

    # Evaluate before storing
    candidate = await memory.remember("ok", skip_store=True)  # Don't store yet

    gate_result = await gate.evaluate(
        memory=candidate,
        existing_memories=all_memories,
    )

    if gate_result.decision.value == "accept":
        await memory.remember(candidate.content)
    # → Only stores high-quality, novel memories

    # ============================================================
    # 6. EVALUATION - Measure performance
    # ============================================================

    from mnemotree.core import SyntheticDatasetGenerator

    evaluator = MemoryEvaluator()
    generator = SyntheticDatasetGenerator()

    # Generate test queries
    eval_queries = generator.generate_queries(all_memories, num_queries=10)

    # Run benchmark
    def retrieval_fn(q):
        return asyncio.run(memory.recall(q, limit=10))

    benchmark = evaluator.evaluate_benchmark(
        queries=eval_queries,
        retrieval_function=retrieval_fn,
        all_memories=all_memories,
    )

    print(f"Recall@5: {benchmark.avg_recall_at_k[5]:.3f}")
    print(f"Precision@5: {benchmark.avg_precision_at_k[5]:.3f}")
    # → Track quality over time


if __name__ == "__main__":
    asyncio.run(quick_start())
