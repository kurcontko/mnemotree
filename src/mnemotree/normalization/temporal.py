"""Temporal anchoring normalizer.

Replaces relative temporal expressions (e.g., "yesterday", "next Monday",
"3 days ago") with absolute ISO-8601 dates, using the reference date
extracted from the content or provided in context.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .base import BaseNormalizer

# Pattern to extract LoCoMo-style date prefix: [1:56 pm on 8 May, 2023]
_LOCOMO_DATE_PREFIX = re.compile(r"^\[([^\]]+)\]")

# Relative temporal expressions to detect and replace.
_RELATIVE_PATTERNS = [
    # "yesterday", "today", "tomorrow"
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
    # "next Monday", "last Friday", etc.
    re.compile(
        r"\b(next|last|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    # "next week", "last month", "next year"
    re.compile(r"\b(next|last|this)\s+(week|month|year)\b", re.IGNORECASE),
    # "N days/weeks/months ago"
    re.compile(r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b", re.IGNORECASE),
    # "in N days/weeks/months"
    re.compile(r"\bin\s+(\d+)\s+(days?|weeks?|months?|years?)\b", re.IGNORECASE),
    # "the day after tomorrow", "the day before yesterday"
    re.compile(r"\bthe\s+day\s+after\s+tomorrow\b", re.IGNORECASE),
    re.compile(r"\bthe\s+day\s+before\s+yesterday\b", re.IGNORECASE),
]


def _parse_reference_date(content: str, context: dict[str, Any] | None = None) -> datetime | None:
    """Extract reference date from content prefix or context."""
    # Try context first
    if context:
        ref = context.get("reference_date")
        if isinstance(ref, datetime):
            return ref
        if isinstance(ref, str):
            return _parse_date_string(ref)

    # Try to parse from LoCoMo-style prefix
    match = _LOCOMO_DATE_PREFIX.match(content)
    if match:
        return _parse_date_string(match.group(1))

    return None


def _parse_date_string(date_str: str) -> datetime | None:
    """Parse a date string using dateparser if available, else common formats."""
    try:
        import dateparser

        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: try common formats
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %B %d, %Y",
        "%d %B, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


class TemporalNormalizer(BaseNormalizer):
    """Anchor relative temporal expressions to absolute dates."""

    async def normalize(self, content: str, context: dict[str, Any] | None = None) -> str:
        ref_date = _parse_reference_date(content, context)
        if ref_date is None:
            return content

        # Find all relative temporal expressions and replace them
        result = content
        for pattern in _RELATIVE_PATTERNS:
            result = self._replace_temporal(result, pattern, ref_date)

        return result

    def _replace_temporal(self, text: str, pattern: re.Pattern, ref_date: datetime) -> str:
        """Replace matches of a temporal pattern with resolved dates."""

        def _replacer(match: re.Match) -> str:
            expr = match.group(0)
            resolved = self._resolve_expression(expr, ref_date)
            if resolved:
                return f"{expr} ({resolved})"
            return expr

        return pattern.sub(_replacer, text)

    @staticmethod
    def _resolve_expression(expr: str, ref_date: datetime) -> str | None:
        """Resolve a relative temporal expression to an ISO date string."""
        try:
            import dateparser

            settings = {"RELATIVE_BASE": ref_date}
            parsed = dateparser.parse(expr, settings=settings)
            if parsed and parsed != ref_date:
                return parsed.strftime("%Y-%m-%d")
        except ImportError:
            pass
        except Exception:
            pass

        return None
