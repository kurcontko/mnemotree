import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chat_ui import MemoryChatUI
from mnemotree.core.memory import MemoryCore
from mnemotree.core.scoring import MemoryScoring
from mnemotree.store.chromadb_store import ChromaMemoryStore

# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


async def init_memory_core() -> MemoryCore:
    """Initialize MemoryEngine with storage."""
    # Initialize store - using ChromaDB with local persistence by default
    store = ChromaMemoryStore(
        persist_directory=".mnemotree/chromadb"
    )
    await store.initialize()

    # Alternative: Neo4j store (requires Neo4j running)
    # try:
    #     store = Neo4jMemoryStore(
    #         uri="bolt://localhost:7687",
    #         user="neo4j",
    #         password="password"
    #     )
    #     await store.initialize()
    # except Exception as e:
    #     print(f"Neo4j store failed to initialize: {e}")
    #     store = ChromaMemoryStore(persist_directory=".mnemotree/chromadb")
    #     await store.initialize()

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    # Tune scoring to favor query relevance and keep more candidates for recall
    memory_scoring = MemoryScoring(
        query_relevance_weight=0.35,
        score_threshold=0.85,
    )

    # Initialize MemoryCore
    memory_core = MemoryCore(
        store=store,
        llm=llm,
        embeddings=embeddings,
        memory_scoring=memory_scoring,
        enable_ner=False  # Disable NER to avoid needing spacy model
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
