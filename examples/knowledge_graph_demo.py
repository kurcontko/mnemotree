"""
Knowledge Graph / Zettelkasten Demo

Demonstrates how to use Mnemotree's knowledge graph features for building
interconnected knowledge bases, similar to Zettelkasten or Roam Research.

Key features demonstrated:
- Creating typed links between memories
- Finding backlinks (what references this?)
- Exploring graph neighborhoods
- Finding connections between concepts

Works in BOTH lite and pro modes - no LLM required!
"""

import asyncio
import tempfile
from pathlib import Path

from mnemotree import LinkType, MemoryCoreBuilder, MemoryType
from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore


async def main():
    # Initialize with SQLite — no external dependencies needed
    db_path = Path(tempfile.mkdtemp()) / "demo.sqlite"
    store = SQLiteVecMemoryStore(db_path=db_path)
    await store.initialize()
    core = MemoryCoreBuilder.lite(store).build()

    print("Mnemotree Knowledge Graph Demo")
    print("=" * 50)
    print()

    # Create a small knowledge base about memory systems
    print("Creating memories...")

    m1 = await core.remember(
        "FSRS (Free Spaced Repetition Scheduler) is an algorithm for optimizing memory retention",
        memory_type=MemoryType.SEMANTIC,
    )
    print(f"  Created: {m1.content[:50]}...")

    m2 = await core.remember(
        "Spaced repetition works by reviewing information at increasing intervals",
        memory_type=MemoryType.SEMANTIC,
    )
    print(f"  Created: {m2.content[:50]}...")

    m3 = await core.remember(
        "The forgetting curve shows that we forget ~50% of new information within 1 hour",
        memory_type=MemoryType.SEMANTIC,
    )
    print(f"  Created: {m3.content[:50]}...")

    m4 = await core.remember(
        "Zettelkasten is a note-taking method that uses interconnected notes",
        memory_type=MemoryType.SEMANTIC,
    )
    print(f"  Created: {m4.content[:50]}...")

    m5 = await core.remember(
        "Roam Research popularized bidirectional links in note-taking apps",
        memory_type=MemoryType.SEMANTIC,
    )
    print(f"  Created: {m5.content[:50]}...")

    print()

    # Create typed links
    print("Creating knowledge graph links...")

    # m2 elaborates on why FSRS works
    link1 = await core.link(
        m2.memory_id,
        m1.memory_id,
        LinkType.ELABORATES,
        context="Explains the mechanism behind FSRS",
    )
    print(f"  {LinkType.ELABORATES.value}: m2 -> m1")

    # m3 supports the need for spaced repetition
    link2 = await core.link(
        m3.memory_id,
        m2.memory_id,
        LinkType.SUPPORTS,
        context="Provides evidence for why spaced repetition is needed",
    )
    print(f"  {LinkType.SUPPORTS.value}: m3 -> m2")

    # m4 and m5 are similar concepts
    link3 = await core.link(
        m4.memory_id,
        m5.memory_id,
        LinkType.SIMILAR_TO,
        context="Both are modern note-taking systems with linked notes",
        bidirectional=True,  # Create reverse link too
    )
    print(f"  {LinkType.SIMILAR_TO.value}: m4 <-> m5 (bidirectional)")

    print()

    # Find backlinks (what references this concept?)
    print("Finding backlinks...")
    print()
    backlinks = await core.get_backlinks(m1.memory_id)
    print(f"Memories that link TO '{m1.content[:40]}...':")
    for mem in backlinks:
        print(f"  <- {mem.content[:60]}...")
    print()

    backlinks_m2 = await core.get_backlinks(m2.memory_id)
    print(f"Memories that link TO '{m2.content[:40]}...':")
    for mem in backlinks_m2:
        print(f"  <- {mem.content[:60]}...")
    print()

    # Get links for a memory
    print("Getting links...")
    links = await core.get_links(m2.memory_id)
    print(f"All links for '{m2.content[:40]}...':")
    for link in links:
        print(f"  {link.link_type.value}: {link.source_id[:8]}... -> {link.target_id[:8]}...")
    print()

    # Explore graph neighborhood
    print("Exploring graph neighborhood...")
    graph = await core.explore(m1.memory_id, depth=2)
    print(f"Graph around '{m1.content[:40]}...':")
    print(f"  Nodes: {graph['node_count']}")
    print(f"  Edges: {graph['edge_count']}")
    print(f"  Nodes found:")
    for node in graph["nodes"]:
        print(f"    - [{node['depth']} hops] {node['content'][:50]}...")
    print()

    # Find connection path
    print("Finding connections...")
    path = await core.find_connection(m3.memory_id, m1.memory_id)
    if path:
        print(f"Path from m3 -> m1:")
        for i, step in enumerate(path, 1):
            print(f"  {i}. {step['memory']['content'][:50]}...")
            print(f"     via {step['link']['type']} link")
            if step["link"]["context"]:
                print(f"     ({step['link']['context']})")
    else:
        print("No connection found")

    print()
    print("=" * 50)
    print("Demo complete!")
    print()
    print("Key takeaways:")
    print("  - Typed links (ELABORATES, SUPPORTS, SIMILAR_TO, etc.)")
    print("  - Bidirectional discovery (backlinks)")
    print("  - Graph traversal and exploration")
    print("  - No LLM required - works in lite mode!")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
