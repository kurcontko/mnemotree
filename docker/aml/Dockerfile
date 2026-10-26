# AML Add/Search service (mnemotree.evidence). Build from the repository root:
#   docker build -f docker/aml/Dockerfile -t mnemotree-aml .
FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# Dependencies install from locked wheels only; no third-party build code runs.
RUN uv sync --frozen --no-dev --no-install-project --no-build --extra serve --extra embeddings
COPY src ./src
# Only this project is built from source, then installed like any other wheel.
RUN uv build --wheel --out-dir /tmp/dist \
    && uv pip install --python .venv --no-deps --no-build /tmp/dist/*.whl

FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data /config && chown app:app /data
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY docker/aml/config.example.json /config/config.json
COPY docker/aml/config.lexical.json /config/config.lexical.json

ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken
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
# The request log (metadata only) is appended under /data/logs, created on startup.
ENTRYPOINT ["mnemotree-evidence"]
CMD ["--db", "/data/aml.sqlite", "--config", "/config/config.json", "serve", "--host", "0.0.0.0", "--port", "8000", "--request-log", "/data/logs/requests.jsonl"]
