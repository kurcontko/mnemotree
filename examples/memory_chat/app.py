import asyncio
import os
import sys

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.memory_chat.chat_ui import MemoryChatUI
from src.core.memory import MemoryCore
from src.store.neo4j_store import Neo4jMemoryStore
from src.store.chromadb_store import ChromaMemoryStore
from src.store.milvus_store import MilvusMemoryStore
from src.store.pgvector_store import PGVectorMemoryStore


async def init_memory_core() -> MemoryCore:
    """Initialize MemoryCore with storage."""
    # Initialize store

    # Neo4j store
    try:
        store = Neo4jMemoryStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mnemosyne_admin"
        )
        await store.initialize()
    except Exception as e:
        print(f"Neo4j store failed to initialize: {e}")
        store = ChromaMemoryStore()
        await store.initialize()

    # Chroma store
    # store = ChromaMemoryStore(
    #     host="localhost",
    #     port=8000,
    # )
    # await store.initialize()

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    # Initialize MemoryCore
    memory_core = MemoryCore(
        llm=llm,
        embeddings=embeddings,
        store=store
    )
    return memory_core


async def init_chat_ui(memory_core: MemoryCore) -> MemoryChatUI:
    """Initialize MemoryChatUI with the injected MemoryCore."""
    chat_ui = MemoryChatUI(memory=memory_core)
    return chat_ui


async def main():
    """Main application entry point."""
    # Initialize MemoryCore
    memory_core = await init_memory_core()

    # Initialize Chat UI with injected MemoryCore
    chat_ui = await init_chat_ui(memory_core)

    # Display the sidebar, chat messages, and process user input
    chat_ui.show_sidebar()
    chat_ui.display_chat_messages()
    await chat_ui.process_user_input()


if __name__ == "__main__":
    asyncio.run(main())
