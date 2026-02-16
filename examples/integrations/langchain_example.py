"""
Example: Using Mnemotree with LangChain

This example shows how to use the LangChainMemoryAdapter to give
LangChain conversations semantic memory with FSRS-4.5 decay.
"""

import asyncio
import os

from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from mnemotree import MemoryCore
from mnemotree.integrations.langchain_adapter import (
    LangChainMemoryAdapter,
    MnemotreeChatMessageHistory,
)
from mnemotree.store import ChromaMemoryStore


async def example_chat_history():
    """Example 1: Using MnemotreeChatMessageHistory for simple chat history."""
    print("=" * 60)
    print("Example 1: Simple Chat History with Decay")
    print("=" * 60)

    # Initialize storage
    store = ChromaMemoryStore(persist_directory=".mnemotree/examples/langchain")
    memory_core = MemoryCore(store=store, mode="lite")  # Lite mode = no API costs

    # Create chat history for a specific session
    history = MnemotreeChatMessageHistory(
        memory_core=memory_core, session_id="conversation-123", max_messages=10
    )

    # Add messages
    await history.aadd_user_message("What's the capital of France?")
    await history.aadd_ai_message("The capital of France is Paris.")
    await history.aadd_user_message("What about Italy?")
    await history.aadd_ai_message("The capital of Italy is Rome.")

    # Retrieve messages (with importance decay applied!)
    messages = await history.aget_messages()
    print(f"\nStored {len(messages)} messages in session")

    for msg in messages:
        print(f"  {msg.type}: {msg.content[:50]}...")

    print("\n✅ Messages stored with automatic importance tracking")


async def example_semantic_memory():
    """Example 2: Using semantic retrieval for context-aware conversations."""
    print("\n" + "=" * 60)
    print("Example 2: Semantic Memory for Conversations")
    print("=" * 60)

    # Initialize
    store = ChromaMemoryStore(persist_directory=".mnemotree/examples/langchain")
    memory_core = MemoryCore(store=store, mode="lite")

    # Create semantic memory adapter
    memory = LangChainMemoryAdapter(
        memory_core=memory_core,
        session_id="user-456",
        return_messages=True,
        max_relevant=5,  # Retrieve top 5 semantically relevant memories
    )

    # Set up LangChain with OpenAI (requires OPENAI_API_KEY)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set, skipping LLM example")
        return

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

    # Have a conversation
    print("\n--- Conversation Start ---")

    response1 = await asyncio.to_thread(
        conversation.predict, input="I really like Python for its readability"
    )
    print("User: I really like Python for its readability")
    print(f"AI: {response1}\n")

    response2 = await asyncio.to_thread(
        conversation.predict, input="What about Rust for systems programming?"
    )
    print("User: What about Rust for systems programming?")
    print(f"AI: {response2}\n")

    # Later conversation - memory will retrieve relevant past context
    response3 = await asyncio.to_thread(
        conversation.predict, input="What programming languages have we discussed?"
    )
    print("User: What programming languages have we discussed?")
    print(f"AI: {response3}\n")

    print("--- Conversation End ---")
    print(
        "\n✅ Conversation uses semantic search to find relevant past interactions"
    )


async def example_decay_demonstration():
    """Example 3: Demonstrate FSRS-4.5 decay in action."""
    print("\n" + "=" * 60)
    print("Example 3: FSRS-4.5 Decay Demonstration")
    print("=" * 60)

    from datetime import datetime, timedelta, timezone

    from mnemotree.core.models import MemoryType

    store = ChromaMemoryStore(persist_directory=".mnemotree/examples/langchain")
    memory_core = MemoryCore(store=store, mode="lite")

    # Store some memories with different types
    episodic = await memory_core.remember(
        "Had a great meeting about the new product launch",
        memory_type=MemoryType.EPISODIC,
        tags=["meeting", "product"],
    )

    semantic = await memory_core.remember(
        "Python uses duck typing for polymorphism",
        memory_type=MemoryType.SEMANTIC,
        tags=["programming", "python"],
    )

    print(f"\nInitial importance:")
    print(f"  Episodic: {episodic.importance:.3f}")
    print(f"  Semantic: {semantic.importance:.3f}")

    # Simulate time passing by manually decaying
    simulated_time = datetime.now(timezone.utc) + timedelta(days=14)  # 14 days later

    episodic.decay_importance(simulated_time)
    semantic.decay_importance(simulated_time)

    print(f"\nAfter 14 days (no access):")
    print(f"  Episodic: {episodic.importance:.3f} (faster decay)")
    print(f"  Semantic: {semantic.importance:.3f} (slower decay)")

    print("\n✅ Different memory types decay at scientifically-determined rates")
    print("   Episodic memories fade faster, semantic knowledge persists longer")


async def main():
    """Run all examples."""
    await example_chat_history()
    await example_semantic_memory()
    await example_decay_demonstration()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
