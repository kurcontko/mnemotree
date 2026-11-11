"""Atom extraction: bounded passage batches, and atoms whose citations are verified quotes.

Atoms are optional derived search records; the original source spans remain the evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import Atom, AtomDraft, Passage, Source, SourceSpan, digest
from ..core.protocols import Extractor
from ..errors import ProviderError


def batches(sources: Sequence[Source], max_chars: int) -> list[list[Passage]]:
    """Bound source text per call, preserving all text and message order."""
    result: list[list[Passage]] = []
    current: list[Passage] = []
    used = 0
    for source in sources:
        for start in range(0, len(source.content), max_chars):
            text = source.content[start : start + max_chars]
            if current and used + len(text) > max_chars:
                result.append(current)
                current, used = [], 0
            current.append(
                Passage(
                    id=digest([source.id, start]),
                    source_id=source.id,
                    start=start,
                    text=text,
                    role=source.role,
                    timestamp=source.timestamp,
                )
            )
            used += len(text)
    if current:
        result.append(current)
    return result


async def extract_atoms(
    extractor: Extractor, sources: Sequence[Source], *, max_chars: int, signature: str
) -> list[Atom]:
    atoms: dict[str, Atom] = {}
    for batch in batches(sources, max_chars):
        passages = {passage.id: passage for passage in batch}
        drafts = list(await extractor.extract(batch))
        if len(drafts) > 64:
            raise ProviderError("extractor returned more than 64 atoms for one batch")
        for value in drafts:
            draft = AtomDraft.model_validate(value)
            ordered = cited_spans(draft, passages)
            atom_id = digest([signature, draft.text, [span.model_dump() for span in ordered]])
            atoms[atom_id] = Atom(
                id=atom_id, text=draft.text, spans=ordered, extractor_signature=signature
            )
    return list(atoms.values())


def cited_spans(draft: AtomDraft, passages: dict[str, Passage]) -> tuple[SourceSpan, ...]:
    """Resolve a draft's citations to verified, deduplicated, sorted source spans."""
    spans: dict[tuple[str, int, int], SourceSpan] = {}
    for citation in draft.citations:
        passage = passages.get(citation.passage_id)
        if passage is None:
            raise ProviderError("atom cites a passage outside its extraction batch")
        offset = passage.text.find(citation.quote)
        if not citation.quote.strip() or offset < 0 or offset != passage.text.rfind(citation.quote):
            raise ProviderError("citation must be an exact, unambiguous quote in its passage")
        start = passage.start + offset
        key = (passage.source_id, start, start + len(citation.quote))
        spans[key] = SourceSpan(source_id=key[0], start=key[1], end=key[2])
    return tuple(spans[key] for key in sorted(spans))
