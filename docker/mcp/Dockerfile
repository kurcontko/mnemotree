FROM python:3.11-slim

WORKDIR /app

# Copy only what is needed
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir uv \
    && uv pip install -e ".[mcp,lite,chroma]"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/chromadb \
    && mkdir -p /app/.mnemotree \
    && chown -R appuser:appuser /app /data

USER appuser

ENV MNEMOTREE_MCP_PERSIST_DIR=/data/chromadb
VOLUME ["/data/chromadb", "/app/.mnemotree"]

EXPOSE 8000
CMD ["fastmcp", "run", "src/mnemotree/mcp/server.py:mcp", "--transport", "http", "--port", "8000"]
