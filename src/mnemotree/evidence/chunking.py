"""Deterministic overlapping source spans; never rewrite the original text."""

from .models import Chunk, digest


def chunk_source(source_id: str, text: str, *, size: int, overlap: int) -> list[Chunk]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("chunk size must be positive and 0 <= overlap < size")
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                id=digest([source_id, start, end]),
                source_id=source_id,
                index=len(chunks),
                start=start,
                end=end,
                text=text[start:end],
            )
        )
        if end == len(text):
            break
        start = end - overlap
    return chunks
