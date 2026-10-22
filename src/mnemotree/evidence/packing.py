"""Exact token accounting and deterministic evidence selection."""

from collections.abc import Sequence
from importlib.metadata import version

import tiktoken

from .models import Evidence
from .protocols import TokenCounter


class TiktokenCounter:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding)
        self.signature = f"tiktoken:{version('tiktoken')}:{encoding}"

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))


def pack(
    evidence: Sequence[Evidence], counter: TokenCounter, *, budget: int, limit: int
) -> tuple[tuple[Evidence, ...], str, int, tuple[str, ...]]:
    selected: list[Evidence] = []
    omitted = []
    context = ""
    tokens = 0
    for item in evidence:
        duplicate = any(
            old.source_id == item.source_id and old.start <= item.start and old.end >= item.end
            for old in selected
        )
        if duplicate or len(selected) >= limit:
            omitted.append(item.id)
            continue
        proposed = "\n\n".join([*(entry.render() for entry in selected), item.render()])
        count = counter.count(proposed)
        if count > budget:
            omitted.append(item.id)
            continue
        selected.append(item)
        context, tokens = proposed, count
    return tuple(selected), context, tokens, tuple(omitted)
