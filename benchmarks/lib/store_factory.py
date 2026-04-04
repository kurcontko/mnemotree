"""Create mnemotree MemoryCore instances for benchmark runs.

Extracted from mnemotree_locomo_optimized.py and evaluate.py so all
benchmark adapters share the same store setup logic.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _ensure_mnemotree_importable() -> None:
    """Add mnemotree src to sys.path if not already installed."""
    workspace_root = Path(__file__).resolve().parents[3]
    mnemotree_root = Path(os.environ.get("MNEMOTREE_ROOT", str(workspace_root / "mnemotree")))
    src = os.environ.get("MNEMOTREE_SRC", str(mnemotree_root / "src"))
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_mnemotree_importable()


async def create_memory_core(
    *,
    store_dir: str | Path,
    sample_id: str | None = None,
    enable_ner: bool = True,
    enable_keywords: bool = True,
    retrieval_mode: str = "hybrid",
    enable_bm25: bool = True,
    reranker: str = "cross_encoder",
    reranker_model: str = "ms-marco-MiniLM-L-6-v2",
    rerank_candidates: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_device: str = "cpu",
    local_models: bool = False,
    router_path: str | None = None,
    extractor_path: str | None = None,
    local_device: str = "cpu",
    clean: bool = True,
) -> Any:
    """Build and return a configured MemoryCore for benchmark use.

    If sample_id is provided, creates an isolated per-sample store directory.
    """
    from mnemotree import MemoryCoreBuilder
    from mnemotree.store import ChromaMemoryStore
    from mnemotree.embeddings.local import LocalSentenceTransformerEmbeddings

    store_path = Path(store_dir)
    if sample_id:
        store_path = store_path / sample_id

    if clean and store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    store_path.mkdir(parents=True, exist_ok=True)

    store = ChromaMemoryStore(
        persist_directory=str(store_path),
        enable_graph_index=enable_ner,
    )
    await store.initialize()

    embeddings = LocalSentenceTransformerEmbeddings(
        model_name=embedding_model,
        device=embedding_device,
    )

    builder = MemoryCoreBuilder(store=store)
    builder = builder.with_mode("lite")
    builder = builder.with_embeddings(embeddings)

    if enable_ner:
        from mnemotree.ner import create_ner
        ner = create_ner("gliner")
        builder = builder.with_ner(ner, enable=True)
    else:
        builder = builder.disable_ner()

    if enable_keywords:
        builder = builder.enable_keywords()
    else:
        builder = builder.disable_keywords()

    if retrieval_mode == "hybrid":
        builder = builder.use_hybrid_fusion(
            reranker_backend=reranker,
            reranker_model=reranker_model,
            rerank_candidates=rerank_candidates,
        )
    else:
        builder = builder.use_basic_retrieval()

    if enable_bm25:
        builder = builder.enable_bm25()

    if local_models:
        builder = builder.with_local_models(
            router_path=router_path,
            extractor_path=extractor_path,
            device=local_device,
        )

    core = builder.build()

    # Explicitly disable normalizer — there's a Heisenbug where coreference
    # resolution gets enabled despite NormalizationConfig defaults. The coref
    # normalizer replaces "I"/"my" with the speaker name ("user"), which
    # garbles content when the speaker is a generic role like "user".
    core.normalizer = None

    return core
