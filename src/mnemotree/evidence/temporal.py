"""Rule-based resolution of relative dates against the message's own timestamp. No LLM.

"I went yesterday" said on 8 May 2023 renders as "... (yesterday = Sunday, 7 May 2023)",
so the reader need not do calendar arithmetic. Week-level phrases resolve to the week's
Monday. On LoCoMo with qwen3.6 this added 13-16 points on temporal questions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import MAXYEAR, MINYEAR, datetime, timedelta

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "couple of": 2,
    "few": 3,
}
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_NUM = r"(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|couple of|few)"
_WEEKDAY = "(" + "|".join(WEEKDAYS) + ")"
# Markers that put a bare "on Saturday" ahead of the message rather than behind it.
_FUTURE = re.compile(r"\b(will|won't|going to|gonna|plan(?:ning)? to|hope to)\b|'ll\b", re.I)
_SENTENCE_END = re.compile(r"[.!?]\s")


def _day(d: datetime) -> str:
    # Names computed directly to stay locale-free.
    weekday = WEEKDAYS[d.weekday()].capitalize()
    return f"{weekday}, {d.day} {_MONTHS[d.month - 1]} {d.year}"


def _week(d: datetime) -> str:
    start = d - timedelta(days=d.weekday())
    return f"week of {start.day} {_MONTHS[start.month - 1]} {start.year}"


def _year(year: int) -> int:
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError(f"year {year} is out of range")
    return year


def _month(d: datetime, shift: int = 0) -> str:
    index = d.month - 1 + shift
    return f"{_MONTHS[index % 12]} {_year(d.year + index // 12)}"


def _n(match: re.Match[str]) -> int:
    value = match.group(1).lower()
    return int(value) if value.isdigit() else NUMBERS[value]


def _weekday(t: datetime, name: str, direction: int) -> datetime:
    target = WEEKDAYS.index(name.lower())
    delta = ((target - t.weekday()) * direction) % 7 or 7
    return t + timedelta(days=delta * direction)


def _future(text: str, start: int) -> bool:
    sentence = _SENTENCE_END.split(text[:start])[-1]
    return _FUTURE.search(sentence) is not None


Rule = Callable[[re.Match[str], datetime, str], str]
RULES: list[tuple[re.Pattern[str], Rule]] = [
    (re.compile(r"\bthe day before yesterday\b", re.I), lambda m, t, s: _day(t - timedelta(2))),
    (re.compile(r"\b(yesterday|last night)\b", re.I), lambda m, t, s: _day(t - timedelta(1))),
    (
        re.compile(r"\b(today|tonight|this (morning|afternoon|evening))\b", re.I),
        lambda m, t, s: _day(t),
    ),
    (re.compile(r"\bthe day after tomorrow\b", re.I), lambda m, t, s: _day(t + timedelta(2))),
    (re.compile(r"\btomorrow\b", re.I), lambda m, t, s: _day(t + timedelta(1))),
    (re.compile(r"\blast (week|weekend)\b", re.I), lambda m, t, s: _week(t - timedelta(7))),
    (re.compile(r"\bthis (week|weekend)\b", re.I), lambda m, t, s: _week(t)),
    (re.compile(r"\bnext (week|weekend)\b", re.I), lambda m, t, s: _week(t + timedelta(7))),
    (re.compile(r"\blast month\b", re.I), lambda m, t, s: _month(t, -1)),
    (re.compile(r"\bthis month\b", re.I), lambda m, t, s: _month(t)),
    (re.compile(r"\bnext month\b", re.I), lambda m, t, s: _month(t, 1)),
    (re.compile(r"\blast year\b", re.I), lambda m, t, s: str(t.year - 1)),
    (re.compile(r"\bthis year\b", re.I), lambda m, t, s: str(t.year)),
    (re.compile(r"\bnext year\b", re.I), lambda m, t, s: str(t.year + 1)),
    (re.compile(rf"\b{_NUM} days? ago\b", re.I), lambda m, t, s: _day(t - timedelta(_n(m)))),
    (
        re.compile(rf"\b{_NUM} weeks? ago\b", re.I),
        lambda m, t, s: _week(t - timedelta(weeks=_n(m))),
    ),
    (re.compile(rf"\b{_NUM} months? ago\b", re.I), lambda m, t, s: _month(t, -_n(m))),
    (re.compile(rf"\b{_NUM} years? ago\b", re.I), lambda m, t, s: str(_year(t.year - _n(m)))),
    (
        re.compile(rf"\b(?:last|this past) {_WEEKDAY}\b", re.I),
        lambda m, t, s: _day(_weekday(t, m.group(1), -1)),
    ),
    (
        re.compile(rf"\bnext {_WEEKDAY}\b", re.I),
        lambda m, t, s: _day(_weekday(t, m.group(1), 1)),
    ),
    (
        re.compile(rf"\bon {_WEEKDAY}\b", re.I),
        lambda m, t, s: _day(_weekday(t, m.group(1), 1 if _future(s, m.start()) else -1)),
    ),
]
# Words every match of the rule at the same index contains (case aside); a rule whose words
# are all absent from a message cannot match, so it is skipped.
_LAST, _THIS, _NEXT = ("last",), ("this",), ("next",)
KEYWORDS: list[tuple[str, ...]] = [
    ("yesterday",),
    ("yesterday", "night"),
    ("today", "tonight", "this"),
    ("tomorrow",),
    ("tomorrow",),
    *(_LAST, _THIS, _NEXT) * 3,
    *(("ago",),) * 4,
    WEEKDAYS,
    WEEKDAYS,
    WEEKDAYS,
]


def resolve(text: str, anchor: datetime | None) -> list[str]:
    """Notes like "yesterday = Sunday, 7 May 2023" for relative phrases, in text order.

    A phrase inside a longer one ("yesterday" in "the day before yesterday") is skipped,
    and each phrase is noted once. A phrase whose date falls outside years 1-9999
    ("1000000 days ago") gets no note; the text itself is kept as written.
    """
    if anchor is None:
        return []
    # For ASCII text a lowercase substring test finds a keyword exactly where re.I would.
    lowered = text.lower() if text.isascii() else None
    found: list[tuple[int, int, str]] = []
    for (pattern, rule), words in zip(RULES, KEYWORDS, strict=True):
        if lowered is not None and not any(word in lowered for word in words):
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < taken_end and taken_start < end for taken_start, taken_end, _ in found):
                continue
            try:
                note = rule(match, anchor, text)
            except (OverflowError, ValueError):
                continue
            found.append((start, end, f"{match.group(0).lower()} = {note}"))
    return list(dict.fromkeys(note for _, _, note in sorted(found)))


def annotate(text: str, anchor: datetime | None) -> str:
    notes = resolve(text, anchor)
    return f"{text} ({'; '.join(notes)})" if notes else text
