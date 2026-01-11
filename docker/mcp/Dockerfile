FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv \
    && uv pip install -e ".[mcp,lite,chroma]"

ENV MNEMOTREE_MCP_PERSIST_DIR=/data/chromadb

EXPOSE 8000
CMD ["fastmcp", "run", "src/mnemotree/mcp/server.py:mcp", "--transport", "http", "--port", "8000"]
