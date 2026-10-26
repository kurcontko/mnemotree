"""A small CLI for local experiments and the optional AML HTTP adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .engine import Memory
from .evaluation import CorpusRecord, ablate, evaluate, read_jsonl
from .models import Record, SearchOptions
from .packing import TiktokenCounter
from .protocols import Embedder


class Config(Record):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    tokenizer: str = "cl100k_base"
    embedding: dict | None = None
    extraction: dict | None = None
    extraction_batch_chars: int = 12000
    search: SearchOptions | None = None


def open_memory(path: Path, config: Config) -> Memory:
    from .atoms import OpenAIExtractor

    extractor = None
    if config.extraction is not None:
        settings = dict(config.extraction)
        if settings.pop("kind") != "openai":
            raise ValueError("extraction kind must be openai")
        if "api_key" in settings:
            raise ValueError("use api_key_env instead of storing a secret in configuration")
        key_env = settings.pop("api_key_env", "EXTRACTION_API_KEY")
        extractor = OpenAIExtractor(api_key=os.environ[key_env], **settings)
    embedder: Embedder | None = None
    if config.embedding is not None:
        from .providers import OpenAIEmbedder, SentenceTransformerEmbedder

        settings = dict(config.embedding)
        kind = settings.pop("kind")
        if kind == "openai":
            if "api_key" in settings:
                raise ValueError("use api_key_env instead of storing a secret in configuration")
            key_env = settings.pop("api_key_env", "EMBEDDING_API_KEY")
            embedder = OpenAIEmbedder(api_key=os.environ[key_env], **settings)
        elif kind == "sentence-transformers":
            embedder = SentenceTransformerEmbedder(**settings)
        else:
            raise ValueError(f"unknown embedding kind: {kind}")
    return Memory(
        path,
        embedder=embedder,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        token_counter=TiktokenCounter(config.tokenizer),
        extractor=extractor,
        extraction_batch_chars=config.extraction_batch_chars,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="JSON settings; defaults to lexical-only")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="ingest native corpus JSONL")
    add.add_argument("corpus", type=Path)
    search = commands.add_parser("search")
    search.add_argument("namespace")
    search.add_argument("query")
    evaluation = commands.add_parser("evaluate", help="source recall; requires an empty database")
    evaluation.add_argument("corpus", type=Path)
    evaluation.add_argument("queries", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    ablation = commands.add_parser(
        "ablate", help="ingest once; compare raw, atoms, and combined retrieval"
    )
    ablation.add_argument("corpus", type=Path)
    ablation.add_argument("queries", type=Path)
    ablation.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve", help="requires the serve extra and a bearer key")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--key-env", default="MNEMOTREE_API_KEY")
    args = parser.parse_args()
    config = Config.model_validate_json(args.config.read_text()) if args.config else Config()
    with open_memory(args.db, config) as memory:
        options = config.search or SearchOptions(mode="hybrid" if memory.embedder else "lexical")
        if args.command == "serve":
            import uvicorn

            from .server import create_app

            uvicorn.run(
                create_app(memory, api_key=os.environ[args.key_env], options=options),
                host=args.host,
                port=args.port,
                access_log=False,
            )
        else:
            asyncio.run(run(args, memory, options))


async def run(args, memory: Memory, options: SearchOptions) -> None:
    if args.command == "add":
        for value in read_jsonl(args.corpus):
            record = CorpusRecord.model_validate(value)
            receipt = await memory.add(
                record.namespace,
                record.messages,
                request_id=record.request_id,
                session_id=record.session_id,
            )
            print(receipt.model_dump_json())
    elif args.command == "search":
        result = await memory.search(args.namespace, args.query, options=options)
        print(result.model_dump_json(indent=2))
    elif args.command in ("evaluate", "ablate"):
        evaluator = ablate if args.command == "ablate" else evaluate
        report = await evaluator(memory, args.corpus, args.queries, options)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
