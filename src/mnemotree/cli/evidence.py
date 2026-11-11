"""A small CLI for local experiments and the optional AML HTTP adapter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from ..config import Config, secret
from ..core.models import SearchOptions
from ..engine import EvidenceEngine
from ..errors import ConfigurationError
from ..evaluation import (
    CorpusRecord,
    ablate,
    ablate_context,
    ablate_retrieval,
    evaluate,
    read_jsonl,
)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive count")
    return number


def configure_logging() -> None:
    """Print the service's JSON lines on stdout; uvicorn configures only its own loggers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    service = logging.getLogger("mnemotree.evidence")
    service.addHandler(handler)
    service.setLevel(logging.INFO)
    service.propagate = False


def code_identity() -> dict[str, str | None]:
    """Identify the loaded mnemotree source even in installed packages without Git.

    A container build may supply MNEMOTREE_BUILD_REVISION as a Git SHA. The source
    digest covers every module of the package and also distinguishes uncommitted
    edits sharing the same revision.
    """
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    revision = os.environ.get("MNEMOTREE_BUILD_REVISION", "")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
        revision = ""
        if (root / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                value = result.stdout.strip()
                if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", value):
                    revision = value
            except (OSError, subprocess.TimeoutExpired):
                # No git binary, or it hung: the source digest below still identifies the code.
                revision = ""
    source_hash = hashlib.sha256()
    try:
        for path in sorted(package.rglob("*.py")):
            source_hash.update(path.relative_to(package).as_posix().encode() + b"\0")
            source_hash.update(path.read_bytes() + b"\0")
        source_digest = source_hash.hexdigest()
    except OSError:
        source_digest = None
    return {"code_revision": revision or None, "source_digest": source_digest}


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
    search.add_argument("--option", action="append", help="an answer option; may be repeated")
    evaluation = commands.add_parser("evaluate", help="source recall; requires an empty database")
    evaluation.add_argument("corpus", type=Path)
    evaluation.add_argument("queries", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--candidate-k", type=positive, default=100)
    ablation = commands.add_parser(
        "ablate", help="ingest once; compare raw, atoms, and combined retrieval"
    )
    ablation.add_argument("corpus", type=Path)
    ablation.add_argument("queries", type=Path)
    ablation.add_argument("--output", type=Path, required=True)
    context_ablation = commands.add_parser(
        "ablate-context", help="ingest once; compare neighboring messages off/on"
    )
    context_ablation.add_argument("corpus", type=Path)
    context_ablation.add_argument("queries", type=Path)
    context_ablation.add_argument("--output", type=Path, required=True)
    retrieval_ablation = commands.add_parser(
        "ablate-retrieval",
        help="ingest once; compare retrieval, role, query, context and budget policies",
    )
    retrieval_ablation.add_argument("corpus", type=Path)
    retrieval_ablation.add_argument("queries", type=Path)
    retrieval_ablation.add_argument("--output", type=Path, required=True)
    retrieval_ablation.add_argument("--candidate-k", type=positive, default=100)
    serve = commands.add_parser("serve", help="requires the serve extra and a bearer key")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--key-env", default="MNEMOTREE_API_KEY")
    # An environment default lets docker compose toggle it without a conditional flag.
    serve.add_argument(
        "--request-log",
        type=Path,
        default=os.environ.get("AML_REQUEST_LOG") or None,
        help="append one metadata-only JSON line per request (default: $AML_REQUEST_LOG)",
    )
    args = parser.parse_args()
    config = Config.model_validate_json(args.config.read_text()) if args.config else Config()
    try:
        api_key = secret(args.key_env) if args.command == "serve" else None
        engine = config.open(args.db)
    except ConfigurationError as exc:
        raise SystemExit(f"mnemotree-evidence: {exc}") from None
    with engine:
        if api_key is not None:
            import uvicorn

            from ..service.app import create_app
            from ..service.presets import aml_options
            from ..service.request_log import BOOT, emit, utc_now

            options = config.search or aml_options(engine)
            app = create_app(
                engine,
                api_key=api_key,
                options=options,
                request_log=args.request_log,
            )
            configure_logging()
            spec = engine.embedder.spec if engine.embedder is not None else None
            emit(
                {
                    "event": "startup",
                    "ts": utc_now(),
                    "boot": BOOT,
                    **code_identity(),
                    "db": str(args.db),
                    "mode": options.mode,
                    "search": options.model_dump(mode="json"),
                    # The spec holds provider, model, dimensions and revision; no key.
                    "embedding": spec.model_dump() if spec is not None else None,
                    "request_log": str(args.request_log) if args.request_log else None,
                },
                args.request_log,
            )
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                access_log=False,
            )
        else:
            options = config.search or engine.default_options()
            asyncio.run(run(args, engine, options))


async def run(args: argparse.Namespace, engine: EvidenceEngine, options: SearchOptions) -> None:
    if args.command == "add":
        for value in read_jsonl(args.corpus):
            record = CorpusRecord.model_validate(value)
            receipt = await engine.add(
                record.namespace,
                record.messages,
                request_id=record.request_id,
                session_id=record.session_id,
            )
            print(receipt.model_dump_json())
    elif args.command == "search":
        result = await engine.search(
            args.namespace, args.query, options=options, query_options=args.option
        )
        print(result.model_dump_json(indent=2))
    elif args.command in ("evaluate", "ablate", "ablate-context", "ablate-retrieval"):
        if args.command in ("evaluate", "ablate-retrieval"):
            evaluator = evaluate if args.command == "evaluate" else ablate_retrieval
            report = await evaluator(
                engine, args.corpus, args.queries, options, candidate_k=args.candidate_k
            )
        else:
            report = await (ablate if args.command == "ablate" else ablate_context)(
                engine, args.corpus, args.queries, options
            )
        report["code_identity"] = code_identity()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
