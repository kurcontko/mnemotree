"""Exact token accounting and deterministic evidence selection."""

import re
from collections.abc import Callable, Sequence
from functools import lru_cache, partial
from importlib.metadata import version
from typing import NamedTuple

import tiktoken

from ..core.models import ContextSpan, Evidence
from ..core.protocols import TokenCounter

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


class Packed(NamedTuple):
    evidence: tuple[Evidence, ...]
    # The joined context, its token count, and the IDs of spans left out of it.
    context: str
    tokens: int
    omitted_ids: tuple[str, ...]


_Rendered = dict[str, tuple[Evidence, str, int]]
_Covered = dict[str, list[tuple[int, int]]]
_OrderKey = Callable[[Evidence], tuple[int, float, int]]
_Fits = Callable[[Sequence[Evidence]], bool]


def pack(
    evidence: Sequence[Evidence],
    counter: TokenCounter,
    *,
    budget: int,
    limit: int,
    merge_reach: int | None = None,
    overflow: Sequence[Evidence] = (),
    order_key: Callable[[Evidence], tuple[int, float, int]] | None = None,
) -> Packed:
    """Select ranked anchors, then add their neighbors while the budget allows.

    ``overflow`` holds lower-priority spans (further chunks of messages already
    anchored); they are placed last, only into budget the anchors and neighbors
    left unused, so a long single-message history is not cut to one chunk.

    Both the joined context (including separators) and the sum of separately
    rendered blocks must fit the budget. ``order_key`` sets presentation order
    before counting; selection still follows relevance. With ``merge_reach``, an
    anchor within that many messages of a selected anchor in the same session
    joins its item instead of using another slot of ``limit``.
    """
    selected: list[Evidence] = []
    covered: _Covered = {}
    omitted: list[str] = []
    by_id = {item.id: item for item in (*evidence, *overflow)}
    rendered: _Rendered = {}
    fits = partial(_fits, counter=counter, budget=budget, order_key=order_key, rendered=rendered)

    # First reserve room for ranked anchors. Neighbor text only uses what
    # remains, so a large adjacent turn cannot evict its matching evidence.
    _place(evidence, selected, omitted, covered, limit, merge_reach, fits)
    for index in range(len(selected)):
        for position, member in enumerate(selected[index].members()):
            _add_neighbors(selected, index, position, by_id[member.id].neighbors, covered, fits)
    # Last, the remaining chunks of already-anchored messages, into unused budget.
    _place(overflow, selected, omitted, covered, limit, merge_reach, fits)
    if order_key is not None:
        selected.sort(key=order_key)
    texts, _ = _render(selected, counter, order_key, rendered)
    context = "\n\n".join(texts)
    return Packed(tuple(selected), context, counter.count(context), tuple(omitted))


def _render(
    items: Sequence[Evidence],
    counter: TokenCounter,
    order_key: _OrderKey | None,
    rendered: _Rendered,
) -> tuple[list[str], int]:
    ordered = sorted(items, key=order_key) if order_key is not None else items
    texts = []
    tokens = 0
    for item in ordered:
        cached = rendered.get(item.id)
        # A merge or added neighbor creates a new immutable block with the same
        # ID. Unchanged blocks keep their rendering and standalone token count.
        if cached is None or cached[0] is not item:
            text = item.render()
            cached = (item, text, counter.count(text))
            rendered[item.id] = cached
        texts.append(cached[1])
        tokens += cached[2]
    return texts, tokens


def _fits(
    items: Sequence[Evidence],
    *,
    counter: TokenCounter,
    budget: int,
    order_key: _OrderKey | None,
    rendered: _Rendered,
) -> bool:
    texts, evidence_tokens = _render(items, counter, order_key, rendered)
    return evidence_tokens <= budget and counter.count("\n\n".join(texts)) <= budget


def _contained(covered: _Covered, source_id: str, start: int, end: int) -> bool:
    return any(low <= start and high >= end for low, high in covered.get(source_id, ()))


def _cover(covered: _Covered, *spans: Evidence | ContextSpan) -> None:
    for span in spans:
        covered.setdefault(span.source_id, []).append((span.start, span.end))


def _touches(block: Evidence, item: Evidence, merge_reach: int | None) -> bool:
    return merge_reach is not None and any(
        member.session_id == item.session_id
        and member.seq is not None
        and item.seq is not None
        and abs(member.seq - item.seq) <= merge_reach
        for member in (block, *block.merged)
    )


def _place(
    items: Sequence[Evidence],
    selected: list[Evidence],
    omitted: list[str],
    covered: _Covered,
    limit: int,
    merge_reach: int | None,
    fits: _Fits,
) -> None:
    for original in items:
        item = original.model_copy(update={"neighbors": ()}) if original.neighbors else original
        if _contained(covered, item.source_id, item.start, item.end):
            omitted.append(item.id)
            continue
        touching = [
            index for index, block in enumerate(selected) if _touches(block, item, merge_reach)
        ]
        if not touching and len(selected) >= limit:
            omitted.append(item.id)
            continue
        proposed = _proposal(selected, touching, item)
        if not fits(proposed):
            omitted.append(item.id)
            continue
        selected[:] = proposed
        _cover(covered, item)


def _proposal(selected: list[Evidence], touching: list[int], item: Evidence) -> list[Evidence]:
    """``selected`` with ``item`` merged into the blocks it touches, or appended."""
    if touching:
        members = [member for index in touching for member in selected[index].members()]
        members.insert(len(selected[touching[0]].members()), item)
        block = members[0].model_copy(update={"merged": tuple(members[1:])})
    else:
        block = item
    proposed = selected.copy()
    for index in reversed(touching[1:]):
        del proposed[index]
    if touching:
        proposed[touching[0]] = block
    else:
        proposed.append(block)
    return proposed


def _add_neighbors(
    selected: list[Evidence],
    index: int,
    position: int,
    neighbors: Sequence[ContextSpan],
    covered: _Covered,
    fits: _Fits,
) -> None:
    for neighbor in neighbors:
        if _contained(covered, neighbor.source_id, neighbor.start, neighbor.end):
            continue
        members = list(selected[index].members())
        members[position] = members[position].model_copy(
            update={"neighbors": (*members[position].neighbors, neighbor)}
        )
        block = members[0].model_copy(update={"merged": tuple(members[1:])})
        proposed = selected.copy()
        proposed[index] = block
        if fits(proposed):
            selected[index] = block
            _cover(covered, neighbor)
