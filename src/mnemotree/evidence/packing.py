"""Exact token accounting and deterministic evidence selection."""

import re
from collections.abc import Sequence
from functools import lru_cache
from importlib.metadata import version

import tiktoken

from .models import ContextSpan, Evidence
from .protocols import TokenCounter

# In these encodings no pre-token runs past a line break followed by anything but
# whitespace or "/", so the pieces between such breaks count exactly as the whole.
LINE_SAFE_ENCODINGS = frozenset({"o200k_base", "cl100k_base"})
_PIECE_BREAK = re.compile(r"(?<=\n)(?=[^\s/])")


class TiktokenCounter:
    # gpt-4o-family encoding; AML's answer model and its tokenizer are unpublished.
    def __init__(self, encoding: str = "o200k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding)
        self.signature = f"tiktoken:{version('tiktoken')}:{encoding}"
        # Packing re-renders a block for every merge and neighbor it tries, so the
        # same lines are counted many times; each piece is encoded once.
        self._piece_count = lru_cache(maxsize=1 << 16)(self._encoded_length)
        self._by_piece = encoding in LINE_SAFE_ENCODINGS

    def _encoded_length(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def count(self, text: str) -> int:
        if not self._by_piece:
            return self._encoded_length(text)
        return sum(map(self._piece_count, _PIECE_BREAK.split(text)))


def pack(
    evidence: Sequence[Evidence],
    counter: TokenCounter,
    *,
    budget: int,
    limit: int,
    merge_reach: int | None = None,
    overflow: Sequence[Evidence] = (),
) -> tuple[tuple[Evidence, ...], str, int, tuple[str, ...]]:
    """Select ranked anchors, then add their neighbors while the budget allows.

    ``overflow`` holds lower-priority spans (further chunks of messages already
    anchored); they are placed last, only into budget the anchors and neighbors
    left unused, so a long single-message history is not cut to one chunk.

    The budget is the sum of each returned item's exact token count: AML
    receives items separately, and counting per item means a change re-counts
    only that item. With ``merge_reach``, an anchor within that many messages
    of a selected anchor in the same session joins its item instead of using
    another slot of ``limit``. Blocks keep the rank of their best anchor.
    """
    selected: list[Evidence] = []
    counts: list[int] = []
    covered: dict[str, list[tuple[int, int]]] = {}
    omitted = []
    by_id = {item.id: item for item in (*evidence, *overflow)}

    def contained(source_id: str, start: int, end: int) -> bool:
        return any(low <= start and high >= end for low, high in covered.get(source_id, ()))

    def cover(*spans: Evidence | ContextSpan) -> None:
        for span in spans:
            covered.setdefault(span.source_id, []).append((span.start, span.end))

    def touches(block: Evidence, item: Evidence) -> bool:
        return merge_reach is not None and any(
            member.session_id == item.session_id
            and member.seq is not None
            and item.seq is not None
            and abs(member.seq - item.seq) <= merge_reach
            for member in (block, *block.merged)
        )

    def place(items: Sequence[Evidence]) -> None:
        for original in items:
            item = original.model_copy(update={"neighbors": ()}) if original.neighbors else original
            if contained(item.source_id, item.start, item.end):
                omitted.append(item.id)
                continue
            touching = [index for index, block in enumerate(selected) if touches(block, item)]
            if not touching and len(selected) >= limit:
                omitted.append(item.id)
                continue
            if touching:
                members = [member for index in touching for member in selected[index].members()]
                members.insert(len(selected[touching[0]].members()), item)
                block = members[0].model_copy(update={"merged": tuple(members[1:])})
            else:
                block = item
            count = counter.count(block.render())
            if sum(counts) - sum(counts[index] for index in touching) + count > budget:
                omitted.append(item.id)
                continue
            for index in reversed(touching[1:]):
                del selected[index], counts[index]
            if touching:
                selected[touching[0]], counts[touching[0]] = block, count
            else:
                selected.append(block)
                counts.append(count)
            cover(item)

    # First reserve room for ranked anchors. Neighbor text only uses what
    # remains, so a large adjacent turn cannot evict its matching evidence.
    place(evidence)
    for index in range(len(selected)):
        for position, member in enumerate(selected[index].members()):
            for neighbor in by_id[member.id].neighbors:
                if contained(neighbor.source_id, neighbor.start, neighbor.end):
                    continue
                members = list(selected[index].members())
                members[position] = members[position].model_copy(
                    update={"neighbors": (*members[position].neighbors, neighbor)}
                )
                block = members[0].model_copy(update={"merged": tuple(members[1:])})
                count = counter.count(block.render())
                if sum(counts) - counts[index] + count <= budget:
                    selected[index], counts[index] = block, count
                    cover(neighbor)
    # Last, the remaining chunks of already-anchored messages, into unused budget.
    place(overflow)
    context = "\n\n".join(entry.render() for entry in selected)
    return tuple(selected), context, sum(counts), tuple(omitted)
