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
    by_id = {item.id: item for item in evidence}
    for original in evidence:
        # First reserve room for ranked anchors. Neighbor text only uses what
        # remains, so a large adjacent turn cannot evict its matching evidence.
        item = original.model_copy(update={"neighbors": ()}) if original.neighbors else original
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
    for index, item in enumerate(selected):
        for neighbor in by_id[item.id].neighbors:
            spans = [(entry.source_id, entry.start, entry.end) for entry in selected]
            spans.extend(
                (span.source_id, span.start, span.end)
                for entry in selected
                for span in entry.neighbors
            )
            if any(
                source_id == neighbor.source_id and start <= neighbor.start and end >= neighbor.end
                for source_id, start, end in spans
            ):
                continue
            expanded = selected[index].model_copy(
                update={"neighbors": (*selected[index].neighbors, neighbor)}
            )
            proposed_items = [*selected[:index], expanded, *selected[index + 1 :]]
            proposed = "\n\n".join(entry.render() for entry in proposed_items)
            count = counter.count(proposed)
            if count <= budget:
                selected[index] = expanded
                context, tokens = proposed, count
    return tuple(selected), context, tokens, tuple(omitted)
