from setuptools import setup, find_packages

# Core requirements - always installed
REQUIRED = [
    "pydantic>=2.0.0,<3.0.0",
    "numpy>=1.26.0,<2.0.0",
    "scikit-learn>=1.6.0,<2.0.0",
    "networkx>=3.4.0,<4.0.0",
    
    # Langchain
    "langchain>=0.3.7,<0.4.0",
    "langchain-openai>=0.3.19,<0.4.0",
    "langchain-community>=0.3.7,<0.4.0",
    "langchain-core>=0.3.19,<0.4.0",
    
    # LlamaIndex
    #"llama-index>=0.12.0,<0.13.0",
    
    # LLMs
    "openai>=1.0.0",
]

# Optional dependencies
EXTRAS = {
    # Individual vector store backends
    "neo4j": ["neo4j>=5.26.0"],
    "milvus": ["pymilvus>=2.5.0"],
    "chroma": ["chromadb>=0.5.23"],
    "qdrant": ["qdrant-client>=1.12.1"],
    "pinecone": ["pinecone-client>=2.2.0"],
    "pgvector": [
        "asyncpg>=0.30.0",
        "psycopg[binary]>=3.2.0"
    ],
    
    # UI dependencies
    "ui": ["streamlit>=1.40.0"],
    
    # Additional LLM providers
    "google": ["langchain-google-genai>=2.0.7"],
    "anthropic": ["langchain-anthropic>=0.3.0"],
    "aws": ["langchain-aws>=0.2.0"],
    "huggingface": ["langchain-huggingface>=0.1.0"],
    
    # Combined sets
    "vectors": [
        "chromadb>=0.5.23",
        "pymilvus>=2.2.0",
        "qdrant-client>=1.12.1",
        "pinecone-client>=2.2.0",
        "asyncpg>=0.30.0",
        "psycopg[binary]>=3.2.0"
    ],
    
    # All dependencies
    "all": [
        "neo4j>=5.26.0",
        "chromadb>=0.5.23",
        "pymilvus>=2.2.0",
        "qdrant-client>=1.12.1",
        "pinecone-client>=2.2.0",
        "asyncpg>=0.30.0",
        "psycopg[binary]>=3.2.0",
        "streamlit>=1.40.0",
        "langchain-google-genai>=2.0.7",
        "langchain-anthropic>=0.3.0",
        "langchain-aws>=0.2.0",
        "langchain-huggingface>=0.1.0"
    ],
}

setup(
    name="mnemotree",
    version="0.1.0",
    description="Advanced memory management and retrieval system for AI Agents",
    author="kurcontko",
    author_email="mikeqrc@gmail.com",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.10",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "mnemotree=mnemotree.cli:main"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Homepage": "https://github.com/kurcontko/mnemotree",
        "Repository": "https://github.com/kurcontko/mnemotree.git",
        "Issues": "https://github.com/kurcontko/mnemotree/issues",
        "Documentation": "https://mnemotree.readthedocs.io",  # Uncomment when docs are ready
    },
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    license="MIT",
    keywords="ai memory management langchain llm agents vector-store",
)