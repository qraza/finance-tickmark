"""Turn the semantic query Claude chose into a retrieval scope.

A "why" question has already been planned as a metric query, so we know its period
and filters. Using them to scope document search is cheap and precise: "why was
marketing under budget in Q2?" searches April-June commentary, with "Marketing"
added to the query, instead of every month.
"""

from __future__ import annotations

from datetime import date


def months_in(grain: str, start: str, end: str | None = None) -> set[str]:
    """All YYYY-MM months covered by a time range, inclusive."""
    s = date.fromisoformat(start)
    last = date.fromisoformat(end or start)
    span = {"month": 1, "quarter": 3, "year": 12}[grain]
    # extend the last period to its final month
    y, m = last.year, last.month + span - 1
    y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    out, cy, cm = set(), s.year, s.month
    while (cy, cm) <= (y, m):
        out.add(f"{cy}-{cm:02d}")
        cy, cm = (cy + 1, 1) if cm == 12 else (cy, cm + 1)
    return out


def scope(semantic_query: dict | None) -> tuple[set[str] | None, list[str]]:
    """(periods, extra query terms) from a semantic query dict; (None, []) if there isn't one."""
    if not semantic_query:
        return None, []
    tr = semantic_query.get("time_range")
    periods = months_in(tr["grain"], tr["start"], tr.get("end")) if tr else None
    terms = [v for f in semantic_query.get("filters", []) if not f.get("negate") for v in f["values"]]
    return periods, terms
