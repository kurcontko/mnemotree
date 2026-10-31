"""Type-check the public API from an installed wheel, outside its source tree."""

from typing import assert_type

from mnemotree import Memory, MemoryItem, Recall


async def check_memory(memory: Memory) -> None:
    item = await memory.remember("An orchid grows here.")
    assert_type(item, MemoryItem)
    recalled = await memory.recall("orchid")
    assert_type(recalled, Recall)
    assert_type(recalled.context, str)
    assert_type(recalled.items, tuple[MemoryItem, ...])
    assert_type(await memory.get(item.id), MemoryItem | None)
    assert_type(await memory.forget(item.id), bool)
