import asyncio
import os
import sys
from typing import List

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.memory import MemoryCore
from src.store.neo4j_store import Neo4jMemoryStore
from src.store.chromadb_store import ChromaMemoryStore
from src.tools.langchain import LangchainMemoryTool


async def init_memory_core() -> MemoryCore:
    """Initialize MemoryCore"""
    # Initialize store
    store = ChromaMemoryStore()
    await store.initialize()

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0
    )

    # Initialize MemoryCore
    memory_core = MemoryCore(
        llm=llm,
        embeddings=embeddings,
        store=store
    )
    return memory_core

async def main():
    # Initialize MemoryCore
    memory_core = await init_memory_core()

    # Initialize LangchainMemoryTool
    memory_tool = LangchainMemoryTool(memory_core)

    # Example of storing initial memories - you'd typically do this based on conversation or events
    await memory_core.remember("My name is Alex.")
    await memory_core.remember("I love playing guitar.")

    # Define tools
    tools: list[Tool] = [
        *memory_tool.get_tools(),
    ]

    # Construct the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You have access to the following tools: {tools}. "
                "Use the tools when appropriate to help answer the user's queries. "
                "You can also remember and recall information to help answer questions better. "
                "Reply in markdown format.",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

     # Format tools into a string to pass to the agent
    formatted_tools = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    # Example interactions
    response = await agent_executor.ainvoke({"input": "What is my name?", "tools": formatted_tools})
    print(f"Response 1: {response['output']}")

    response = await agent_executor.ainvoke({"input": "What are my hobbies?", "tools": formatted_tools})
    print(f"Response 2: {response['output']}")

    response = await agent_executor.ainvoke({"input": "What is the capital of France?", "tools": formatted_tools})
    print(f"Response 3: {response['output']}")

if __name__ == "__main__":
    import asyncio
    
    asyncio.run(main())
