"""Candidate row IDs shared by FTS5 and sqlite-vec when a search excludes evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ..core.models import SourceSpan


@dataclass(frozen=True)
class RowFilter:
    query: str = ""
    params: tuple[str, ...] = ()

    def condition(self, column: str) -> str:
        # The column and query come only from the store. All values are bound.
        return f" AND {column} IN ({self.query})" if self.query else ""

    def atoms(self, namespace: str) -> RowFilter:
        if not self.query:
            return self
        return RowFilter(
            "SELECT DISTINCT a.rowid FROM evidence_atoms a JOIN evidence_atom_spans s "
            "ON s.namespace=a.namespace AND s.atom_id=a.id JOIN evidence_chunks c "
            "ON c.namespace=s.namespace AND c.source_id=s.source_id "
            "AND c.start<s.end AND c.end>s.start WHERE a.namespace=?" + self.condition("c.rowid"),
            (namespace, *self.params),
        )


ALL_ROWS = RowFilter()


def eligible_chunks(
    namespace: str,
    exclude_session: str | None,
    spans: Sequence[SourceSpan],
    since: datetime | None = None,
) -> RowFilter:
    """Every chunk except the excluded session's and the excluded spans.

    With ``since``, the session loses only its messages from that time on. A message
    without a timestamp cannot be placed, so it stays excluded.
    """
    excluded = []
    params = [namespace]
    if exclude_session is not None:
        excluded.append(
            "SELECT c.rowid FROM evidence_sources s JOIN evidence_chunks c "
            "ON c.namespace=s.namespace AND c.source_id=s.id WHERE s.namespace=? AND s.session_id=?"
        )
        params.extend((namespace, exclude_session))
        if since is not None:
            # julianday() reads the stored ISO text with its offset; text order would not.
            excluded[-1] += " AND (s.timestamp IS NULL OR julianday(s.timestamp)>=julianday(?))"
            params.append(since.isoformat())
    if spans:
        excluded.append(
            "SELECT c.rowid FROM json_each(?) r JOIN evidence_chunks c "
            "ON c.namespace=? AND c.source_id=json_extract(r.value,'$[0]') "
            "AND c.start>=json_extract(r.value,'$[1]') AND c.end<=json_extract(r.value,'$[2]')"
        )
        params.extend((json.dumps([(s.source_id, s.start, s.end) for s in spans]), namespace))
    if not excluded:
        return ALL_ROWS
    return RowFilter(
        "SELECT rowid FROM evidence_chunks WHERE namespace=? AND rowid NOT IN ("
        + " UNION ALL ".join(excluded)
        + ")",
        tuple(params),
    )
