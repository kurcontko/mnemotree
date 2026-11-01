"""Embedding vectors as the float32 blobs sqlite-vec stores."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence


def vector_blob(values: Sequence[float], dimensions: int) -> bytes:
    """Validate and unit-normalize ``values`` as a little-endian float32 blob."""
    if len(values) != dimensions:
        raise ValueError(f"expected {dimensions} embedding dimensions, got {len(values)}")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    try:
        blob = struct.pack(f"<{dimensions}f", *vector)
    except (OverflowError, struct.error) as exc:
        raise ValueError("embedding must fit float32") from exc
    floats = struct.unpack(f"<{dimensions}f", blob)
    if not all(math.isfinite(value) for value in floats) or not any(floats):
        raise ValueError("cosine embeddings must be finite and nonzero in float32")
    # sqlite-vec accumulates cosine products in float32. Unit-normalizing here
    # prevents finite large values from overflowing those products.
    norm = math.hypot(*floats)
    return struct.pack(f"<{dimensions}f", *(value / norm for value in floats))
