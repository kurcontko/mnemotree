"""Pure ranking steps of retrieval: query bounding, fusion and ordering policies.

Nothing here touches the store or a provider, so each step is tested on plain values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .models import Candidate, ChannelScore, Chunk, Evidence, SearchOptions
from .protocols import TokenCounter

# Cap the final model input with the configured tokenizer. The service tokenizer
# is a proxy for the provider's tokenizer, so keep margin below its 8,192 limit.
QUERY_EMBED_TOKENS = 6000
QUERY_EMBED_CHARS = 6000


def model_query(
    query: str, limit: int = QUERY_EMBED_CHARS, counter: TokenCounter | None = None
) -> str:
    """Preserve small inputs; otherwise retain endpoints within the final input cap."""
    separator = "\n...\n"

    def endpoints(chars: int) -> str:
        if chars < len(separator):
            return query[:chars]
        retained = chars - len(separator)
        head = (retained + 1) // 2
        tail = retained // 2
        return query[:head] + separator + (query[-tail:] if tail else "")

    if counter is None:
        return query if len(query) <= limit else endpoints(max(0, limit))
    if counter.count(query) <= QUERY_EMBED_TOKENS:
        return query
    # Count every candidate, including its separator. Character/token ratios can
    # differ radically between the middle and endpoints, even in short inputs.
    low, high = 0, len(query) - 1
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = endpoints(middle)
        if counter.count(candidate) <= QUERY_EMBED_TOKENS:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def channel_weights(options: SearchOptions) -> dict[str, float]:
    """RRF weight of each channel; atom channels scale their raw counterpart."""
    return {
        "lexical": options.lexical_weight,
        "dense": options.dense_weight,
        "atom_lexical": options.lexical_weight * options.atom_weight,
        "atom_dense": options.dense_weight * options.atom_weight,
    }


def fuse(
    channels: dict[str, list[tuple[Chunk, float]]],
    *,
    weights: dict[str, float],
    k: int,
    ranks: dict[str, dict[str, int]] | None = None,
) -> list[Candidate]:
    """Weighted reciprocal rank fusion; ``ranks`` overrides a channel's list position."""
    by_id: dict[str, Chunk] = {}
    scores: dict[str, dict[str, ChannelScore]] = {}
    for channel, items in channels.items():
        for rank, (chunk, score) in enumerate(items, 1):
            by_id[chunk.id] = chunk
            source_rank = (ranks or {}).get(channel, {}).get(chunk.id, rank)
            scores.setdefault(chunk.id, {})[channel] = ChannelScore(rank=source_rank, score=score)
    candidates = [
        Candidate(
            chunk=by_id[id],
            channels=contributions,
            fusion_score=sum(
                weights[channel] / (k + value.rank) for channel, value in contributions.items()
            ),
        )
        for id, contributions in scores.items()
    ]
    return sorted(candidates, key=by_fusion)


def by_fusion(candidate: Candidate) -> tuple[float, str]:
    return (-candidate.fusion_score, candidate.chunk.id)


def by_rerank(candidate: Candidate) -> tuple[float, float, str]:
    return (-(candidate.rerank_score or 0.0), -candidate.fusion_score, candidate.chunk.id)


def cap_per_source(
    candidates: Sequence[Candidate], per_source: int
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into the best `per_source` of each source message and the rest.

    Both lists keep the given order. The rest is not discarded: the packer adds it
    once anchors and their neighbors have been placed, so a history that is one long
    document is not served as a single chunk of it.
    """
    seen: dict[str, int] = {}
    kept: list[Candidate] = []
    overflow: list[Candidate] = []
    for candidate in candidates:
        count = seen.get(candidate.chunk.source_id, 0)
        (kept if count < per_source else overflow).append(candidate)
        seen[candidate.chunk.source_id] = count + 1
    return kept, overflow


class RolePolicy(NamedTuple):
    """Which role policy a search applies, after the assistant-share gate."""

    boost: bool
    partition: bool


def role_policy(options: SearchOptions, assistant_share: float | None) -> RolePolicy:
    enabled = options.role_first is not None and (
        options.role_first_min_assistant_share is None
        or (
            assistant_share is not None
            and assistant_share >= options.role_first_min_assistant_share
        )
    )
    return RolePolicy(
        boost=enabled and options.role_boost is not None,
        partition=enabled and options.role_boost is None,
    )


def boost_role(
    candidates: Sequence[Candidate], roles: Mapping[str, str], role: str, boost: float
) -> list[Candidate]:
    """Add ``boost`` to the fusion score of candidates whose source has ``role``."""
    return sorted(
        (
            item.model_copy(
                update={
                    "fusion_score": item.fusion_score
                    + (boost if roles.get(item.chunk.id) == role else 0.0)
                }
            )
            for item in candidates
        ),
        key=by_fusion,
    )


def role_first(
    candidates: Sequence[Candidate], roles: Mapping[str, str], role: str
) -> list[Candidate]:
    """Stable partition: candidates whose source has `role` first, then the rest."""
    first = [c for c in candidates if roles.get(c.chunk.id) == role]
    rest = [c for c in candidates if roles.get(c.chunk.id) != role]
    return first + rest


def chronological(item: Evidence) -> tuple[int, float, int]:
    """Earliest member timestamp, then session order; untimed items follow timed ones."""
    members = item.members()
    stamps = [member.timestamp.timestamp() for member in members if member.timestamp is not None]
    seq = min((member.seq for member in members if member.seq is not None), default=0)
    return (0, min(stamps), seq) if stamps else (1, 0.0, seq)
