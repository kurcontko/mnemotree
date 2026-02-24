"""Coreference resolution normalizer.

Resolves pronouns in memory content to their referents so that stored text
contains proper names instead of ambiguous pronouns like "he", "she", "they".

Two backends:
  - "heuristic" (default): regex-based, no extra dependencies
  - "fastcoref": neural coreference using biu-nlp/f-coref model
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseNormalizer

# Patterns for first-person pronoun replacement (speaker → speaker name).
# Each tuple: (pattern, replacement template) where {name} is substituted.
_FIRST_PERSON = [
    # Standalone words only (word boundaries)
    (re.compile(r"\bI\b(?!')"), "{name}"),
    (re.compile(r"\bme\b", re.IGNORECASE), "{name}"),
    (re.compile(r"\bmy\b", re.IGNORECASE), "{name}'s"),
    (re.compile(r"\bmine\b", re.IGNORECASE), "{name}'s"),
    (re.compile(r"\bmyself\b", re.IGNORECASE), "{name}"),
]

# Third-person singular pronouns (mapped to the other speaker).
_THIRD_PERSON_PATTERNS = [
    (re.compile(r"\b[Hh]e\b"), "{name}"),
    (re.compile(r"\b[Ss]he\b"), "{name}"),
    (re.compile(r"\b[Hh]im\b"), "{name}"),
    (re.compile(r"\b[Hh]er\b(?!\s+\w+ing)"), "{name}"),  # avoid "her doing"
    (re.compile(r"\b[Hh]is\b"), "{name}'s"),
    (re.compile(r"\b[Hh]ers\b"), "{name}'s"),
    (re.compile(r"\b[Hh]imself\b"), "{name}"),
    (re.compile(r"\b[Hh]erself\b"), "{name}"),
]

# Pattern to extract speaker and text from LoCoMo-style formatted content:
# "[date] Speaker: text"
_SPEAKER_PATTERN = re.compile(r"^\[.*?\]\s*(\w[\w\s]*?):\s*(.*)$", re.DOTALL)


class CoreferenceNormalizer(BaseNormalizer):
    """Resolve coreferences (pronouns) in memory content.

    Args:
        backend: "heuristic" for regex-based resolution (default, no deps),
                 "fastcoref" for neural resolution (requires fastcoref package).
    """

    def __init__(self, backend: str = "heuristic") -> None:
        self.backend = backend
        self._fastcoref_model: Any = None

    async def normalize(self, content: str, context: dict[str, Any] | None = None) -> str:
        if self.backend == "fastcoref":
            return self._normalize_fastcoref(content, context)
        return self._normalize_heuristic(content, context)

    # ------------------------------------------------------------------
    # Heuristic backend
    # ------------------------------------------------------------------

    def _normalize_heuristic(self, content: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        speaker = ctx.get("speaker")
        other_speaker = ctx.get("other_speaker")

        # Try to parse speaker from content format "[date] Speaker: text"
        match = _SPEAKER_PATTERN.match(content)
        if match and not speaker:
            speaker = match.group(1).strip()

        if not speaker:
            return content

        # Extract the text portion after "Speaker:"
        if match:
            prefix = content[: match.start(2)]
            text = match.group(2)
        else:
            prefix = ""
            text = content

        # Replace first-person pronouns with speaker name
        for pattern, template in _FIRST_PERSON:
            text = pattern.sub(template.format(name=speaker), text)

        # Replace third-person pronouns with other_speaker if unambiguous
        if other_speaker:
            # Only replace if there's a single candidate referent
            recent_entities = ctx.get("recent_entities", [])
            # If we have recent entities, check for ambiguity
            person_candidates = {other_speaker}
            for entity in recent_entities:
                if isinstance(entity, str) and entity != speaker:
                    person_candidates.add(entity)

            # Only resolve if unambiguous (single candidate)
            if len(person_candidates) == 1:
                for pattern, template in _THIRD_PERSON_PATTERNS:
                    text = pattern.sub(template.format(name=other_speaker), text)

        return prefix + text

    # ------------------------------------------------------------------
    # fastcoref backend
    # ------------------------------------------------------------------

    def _normalize_fastcoref(self, content: str, context: dict[str, Any] | None = None) -> str:
        try:
            from fastcoref import FCoref
        except ImportError:
            # Fall back to heuristic if fastcoref not installed
            return self._normalize_heuristic(content, context)

        if self._fastcoref_model is None:
            self._fastcoref_model = FCoref()

        preds = self._fastcoref_model.predict(texts=[content])
        resolved = preds[0].get_resolved_text()
        return resolved if resolved else content
