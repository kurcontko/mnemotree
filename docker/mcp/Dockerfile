# MCP stdio server with the local embedding model baked in (mnemotree-mcp).
# Build from the repository root:
#   docker build -f docker/mcp/Dockerfile -t mnemotree-mcp .
# An MCP client starts one container per session and talks to it over stdin/stdout:
#   docker run -i --rm --network none -v mnemotree-memory:/data mnemotree-mcp
FROM python:3.14-slim-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY packages/mnemotree-mcp/pyproject.toml packages/mnemotree-mcp/README.md packages/mnemotree-mcp/
# Dependencies install from locked wheels only; no third-party build code runs.
RUN uv sync --frozen --no-dev --no-install-workspace --no-build --package mnemotree-mcp
COPY src ./src
COPY packages/mnemotree-mcp/src packages/mnemotree-mcp/src
# Only the two workspace projects are built from source, then installed as wheels.
RUN uv build --wheel --all-packages --out-dir /tmp/dist \
    && uv pip install --python .venv --no-deps --no-build /tmp/dist/*.whl

FROM python:3.14-slim-trixie

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app:app /data
COPY --from=build --chown=app:app /app/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    HF_HUB_OFFLINE=1 \
    MNEMOTREE_DB=/data/memory.sqlite
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" && chown app:app "$TIKTOKEN_CACHE_DIR"

USER app
# Bake the tokenizer and the pinned embedding model (about 200 MB) into
# ~/.mnemotree/models, so a session starts at once and needs no network.
RUN HF_HUB_OFFLINE=0 python -c "\
import tiktoken; tiktoken.get_encoding('o200k_base'); \
from mnemotree.config import Config; \
from mnemotree.providers.embeddings import DEFAULT_LOCAL_EMBEDDING; \
Config.model_validate({'embedding': DEFAULT_LOCAL_EMBEDDING}).embedding.build().warm()"

VOLUME ["/data"]
# stdout carries the protocol; logs go to stderr.
ENTRYPOINT ["mnemotree-mcp"]
