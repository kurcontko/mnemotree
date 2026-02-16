"""
Example: Using Mnemotree with LlamaIndex

This example shows how to use the LlamaIndexMemoryAdapter to give
LlamaIndex chat engines semantic memory with FSRS-4.5 decay.
"""

import asyncio
import os


async def example_vector_memory():
    """Example: Using MnemotreeVectorMemory with LlamaIndex chat engine."""
    print("=" * 60)
    print("Example: LlamaIndex Chat Engine with Mnemotree Memory")
    print("=" * 60)

    try:
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.llms.openai import OpenAI
    except ImportError:
        print("⚠️  LlamaIndex not installed. Install with:")
        print("    pip install llama-index llama-index-llms-openai")
        return

    from mnemotree import MemoryCore
    from mnemotree.integrations.llamaindex_adapter import MnemotreeVectorMemory
    from mnemotree.store import ChromaMemoryStore

    # Initialize
    store = ChromaMemoryStore(persist_directory=".mnemotree/examples/llamaindex")
    memory_core = MemoryCore(store=store, mode="lite")

    # Create LlamaIndex-compatible memory
    memory = MnemotreeVectorMemory(
        memory_core=memory_core, user_id="user-789", max_messages=10
    )

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set, running without LLM")
        print("\nDemonstrating memory storage and retrieval:")

        # Store messages directly
        await memory.aput(
            ChatMessage(
                role=MessageRole.USER, content="I'm interested in machine learning"
            )
        )
        await memory.aput(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Great! Machine learning is a fascinating field. What aspect interests you most?",
            )
        )
        await memory.aput(
            ChatMessage(role=MessageRole.USER, content="Neural networks specifically")
        )

        # Retrieve memories
        messages = await memory.aget(query="machine learning")
        print(f"\nRetrieved {len(messages)} relevant messages:")
        for msg in messages:
            print(f"  {msg.role.value}: {msg.content[:60]}...")

        print("\n✅ Messages stored and retrieved using semantic search")
        return

    # Full example with LLM
    llm = OpenAI(model="gpt-4", temperature=0)

    print("\n--- Starting Conversation ---")

    # Create chat engine
    chat_engine = SimpleChatEngine.from_defaults(llm=llm)

    # First conversation
    response1 = await chat_engine.achat("I'm learning about distributed systems")
    print(f"\nUser: I'm learning about distributed systems")
    print(f"AI: {response1.response}\n")

    # Store in memory
    await memory.aput(
        ChatMessage(role=MessageRole.USER, content="I'm learning about distributed systems")
    )
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=response1.response))

    # Second conversation
    response2 = await chat_engine.achat("What are the main challenges?")
    print(f"User: What are the main challenges?")
    print(f"AI: {response2.response}\n")

    await memory.aput(ChatMessage(role=MessageRole.USER, content="What are the main challenges?"))
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=response2.response))

    # Later - retrieve relevant context
    print("\n--- Later Session ---")
    relevant_memories = await memory.aget(query="distributed systems")

    print(f"\nRetrieved {len(relevant_memories)} relevant past messages:")
    for msg in relevant_memories:
        print(f"  {msg.role.value}: {msg.content[:80]}...")

    # Use retrieved context for new question
    context_str = "\n".join([f"{m.role.value}: {m.content}" for m in relevant_memories])

    response3 = await chat_engine.achat(
        f"Based on our previous discussion:\n{context_str}\n\nCan you summarize what I've learned?"
    )
    print(f"\nUser: Can you summarize what I've learned?")
    print(f"AI: {response3.response}\n")

    print("--- Conversation End ---")
    print("\n✅ Chat engine with persistent, semantic memory")


async def example_memory_decay():
    """Example: Demonstrate how memory importance decays over time."""
    print("\n" + "=" * 60)
    print("Example: Memory Decay with LlamaIndex")
    print("=" * 60)

    from datetime import datetime, timedelta, timezone

    from mnemotree import MemoryCore
    from mnemotree.core.models import MemoryType
    from mnemotree.integrations.llamaindex_adapter import MnemotreeVectorMemory
    from mnemotree.store import ChromaMemoryStore

    store = ChromaMemoryStore(persist_directory=".mnemotree/examples/llamaindex")
    memory_core = MemoryCore(store=store, mode="lite")

    # Store different types of memories
    important_fact = await memory_core.remember(
        "The speed of light is 299,792,458 meters per second",
        memory_type=MemoryType.SEMANTIC,
        tags=["physics", "constants"],
        importance=0.9,
    )

    casual_note = await memory_core.remember(
        "Had coffee with John this morning",
        memory_type=MemoryType.EPISODIC,
        tags=["social"],
        importance=0.5,
    )

    print("\nInitial state:")
    print(f"  Important fact: importance={important_fact.importance:.3f}")
    print(f"  Casual note: importance={casual_note.importance:.3f}")

    # Simulate 30 days passing
    simulated_time = datetime.now(timezone.utc) + timedelta(days=30)

    important_fact.decay_importance(simulated_time)
    casual_note.decay_importance(simulated_time)

    print("\nAfter 30 days (no access):")
    print(f"  Important fact: importance={important_fact.importance:.3f} (semantic = slow decay)")
    print(f"  Casual note: importance={casual_note.importance:.3f} (episodic = fast decay)")

    # Simulate accessing the important fact
    important_fact.update_access()
    print("\nAfter accessing important fact:")
    print(f"  Important fact: importance={important_fact.importance:.3f} (reinforced!)")

    print(
        "\n✅ FSRS-4.5 decay ensures important, frequently-accessed memories persist"
    )


async def main():
    """Run all examples."""
    await example_vector_memory()
    await example_memory_decay()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
