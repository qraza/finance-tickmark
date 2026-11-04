"""Turn the semantic query Claude chose into a retrieval scope.

A "why" question has already been planned as a metric query, so we know its period
and filters. Using them to scope document search is cheap and precise: "why was
marketing under budget in Q2?" searches April-June commentary, with "Marketing"
added to the query, instead of every month.
"""

from __future__ import annotations

from datetime import date


def months_in(grain: str, start: str, end: str | None = None) -> set[str]:
    """All YYYY-MM months a time range covers, read exactly as the query's filter reads it.

    Both ends snap to the start of their period, then every period in between is
    expanded to its months: quarter 2026-07-01..2026-09-01 is Q3, July to September.
    """
    from ..semantic.query import period_start

    span = {"month": 1, "quarter": 3, "year": 12}[grain]
    s = date.fromisoformat(period_start(grain, start))
    e = date.fromisoformat(period_start(grain, end or start))
    out, y, m = set(), s.year, s.month
    last = (e.year, e.month + span - 1) if e.month + span - 1 <= 12 else (e.year + 1, e.month + span - 13)
    while (y, m) <= last:
        out.add(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def scope(semantic_query: dict | None) -> tuple[set[str] | None, list[str]]:
    """(periods, extra query terms) from a semantic query dict; (None, []) if there isn't one."""
    if not semantic_query:
        return None, []
    tr = semantic_query.get("time_range")
    periods = months_in(tr["grain"], tr["start"], tr.get("end")) if tr else None
    terms = [v for f in semantic_query.get("filters", []) if not f.get("negate") for v in f["values"]]
    return periods, terms
