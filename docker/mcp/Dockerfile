FROM python:3.11-slim

WORKDIR /app

# Copy only what is needed
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir uv && \
    uv pip install --system -e ".[mcp_server,lite,sqlite_vec]" && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    mkdir -p /data && \
    mkdir -p /app/.mnemotree && \
    chown -R appuser:appuser /app /data

USER appuser

ENV MNEMOTREE_MCP_STORE_BACKEND=sqlite
ENV MNEMOTREE_MCP_PERSIST_DIR=/data/mnemotree.sqlite
VOLUME ["/data"]

EXPOSE 8000
CMD ["fastmcp", "run", "src/mnemotree/mcp/server.py:mcp", "--transport", "http", "--port", "8000"]
