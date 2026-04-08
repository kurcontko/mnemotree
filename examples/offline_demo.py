#!/usr/bin/env python3
"""Offline demo — no API keys, no cloud services, no LLM required.

Requirements:
    pip install mnemotree sentence-transformers

This demo stores a few memories using a local sentence-transformer model
and an SQLite database, then retrieves them by semantic similarity.
Everything runs on your machine — nothing leaves localhost.
"""

import asyncio
import tempfile
from pathlib import Path


async def main() -> None:
    from mnemotree.core.builder import MemoryCoreBuilder
    from mnemotree.embeddings.local import LocalSentenceTransformerEmbeddings

    # 1. Set up local embeddings (downloads ~90 MB model on first run)
    embedder = LocalSentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # 2. Build MemoryCore with a temp SQLite database
    db_path = Path(tempfile.mkdtemp()) / "demo.sqlite"
    core = MemoryCoreBuilder.default(db_path=db_path).with_embeddings(embedder).build()
    await core.store.initialize()

    # 3. Store some memories
    await core.remember("Python's GIL prevents true parallelism in CPU-bound threads")
    await core.remember("asyncio provides cooperative concurrency for I/O-bound work")
    await core.remember("The Eiffel Tower was completed in 1889 for the World's Fair")
    await core.remember("Rust's ownership model eliminates data races at compile time")
    print(f"Stored 4 memories in {db_path}")

    # 4. Recall by semantic similarity
    results = await core.recall("How does Python handle concurrency?", limit=2)
    print("\nQuery: 'How does Python handle concurrency?'")
    for i, mem in enumerate(results, 1):
        print(f"  {i}. {mem.content}")

    # 5. Knowledge graph: link related memories
    all_mems = await core.recall("programming", limit=4)
    if len(all_mems) >= 2:
        from mnemotree.core.models import LinkType

        link = await core.link(
            all_mems[0].memory_id,
            all_mems[1].memory_id,
            LinkType.SIMILAR_TO,
        )
        print(f"\nLinked: {all_mems[0].content[:40]}... -> {all_mems[1].content[:40]}...")
        print(f"  Link ID: {link.link_id}")

    await core.store.close()
    print("\nDone! No API keys were used.")


if __name__ == "__main__":
    asyncio.run(main())
