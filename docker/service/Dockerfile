# Add/Search HTTP service (mnemotree.service). Build from the repository root:
#   docker build -f docker/service/Dockerfile -t mnemotree-service .
FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY packages/mnemotree-mcp/pyproject.toml packages/mnemotree-mcp/pyproject.toml
# Dependencies install from locked wheels only; no third-party build code runs.
RUN uv sync --frozen --no-dev --no-install-workspace --no-build --extra serve --extra embeddings
COPY src ./src
# Only this project is built from source, then installed like any other wheel.
RUN uv build --wheel --out-dir /tmp/dist \
    && uv pip install --python .venv --no-deps --no-build /tmp/dist/*.whl

FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app:app /data
COPY --from=build --chown=app:app /app/.venv /app/.venv

# The startup log records this commit; the image has no Git metadata.
ARG MNEMOTREE_BUILD_REVISION=""
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    MNEMOTREE_BUILD_REVISION=$MNEMOTREE_BUILD_REVISION
# Bake tokenizer files so startup never depends on the network.
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]" \
    && chown -R app:app "$TIKTOKEN_CACHE_DIR"

USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One process only: the SQLite store serialises access behind a single lock.
# Without --config the service is lexical. For embeddings, mount a config and
# put `--config /config/config.json` before `serve`.
ENTRYPOINT ["mnemotree-evidence"]
CMD ["--db", "/data/memory.sqlite", "serve", "--host", "0.0.0.0", "--port", "8000"]
