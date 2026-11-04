"""Check that the period Claude queried is the period the question asked about.

The first pipeline run caught Claude answering "furthest under budget in Q3" with
year-to-date figures. This module reads an explicit period from the question (Q3,
August, YTD, this quarter) and compares it with the plan's time range. A mismatch
triggers one re-plan with the mismatch spelt out; a second mismatch is refused.

Only unambiguous cases are checked: one period named in the question. "Q2 vs Q3" or
no period at all is left to the model.
"""

from __future__ import annotations

import re

from .rag.scope import months_in

YEAR = 2026  # the only fiscal year in the data; a question without a year means this one
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
# Month names must be capitalised ("May", not "may"), so modal verbs don't count.
_MONTH = re.compile(r"\b(" + "|".join(MONTHS + [m[:3] for m in MONTHS if m != "May"]) + r")\b(?:\s+(\d{4}))?")
_QUARTER = re.compile(r"\bQ([1-4])\b(?:\s+(\d{4}))?", re.IGNORECASE)
_YTD = re.compile(r"\b(ytd|year[- ]to[- ]date)\b", re.IGNORECASE)
_THIS_Q = re.compile(r"\b(this|latest|current|last closed) quarter\b", re.IGNORECASE)


def asked_months(question: str) -> set[str] | None:
    """The months a question explicitly asks about, or None if it names none or several."""
    found: list[set[str]] = []
    for q, y in _QUARTER.findall(question):
        found.append(months_in("quarter", f"{y or YEAR}-{3 * (int(q) - 1) + 1:02d}-01"))
    for name, y in _MONTH.findall(question):
        idx = next(i for i, m in enumerate(MONTHS) if m.startswith(name))
        found.append({f"{y or YEAR}-{idx + 1:02d}"})
    if _YTD.search(question):
        found.append(months_in("month", f"{YEAR}-01-01", f"{YEAR}-09-01"))
    if _THIS_Q.search(question):
        found.append(months_in("quarter", f"{YEAR}-07-01"))
    return found[0] if len(found) == 1 else None


def mismatch(question: str, time_range: dict | None) -> str | None:
    """A message describing the mismatch, or None if the plan's period fits the question."""
    want = asked_months(question)
    if want is None:
        return None
    got = months_in(time_range["grain"], time_range["start"], time_range.get("end")) if time_range else set()
    if got == want:
        return None
    return (f"The question asks about {_describe(want)}, but your query covered "
            f"{_describe(got) if got else 'no time range'}. Set the time range to match the question.")


def _describe(months: set[str]) -> str:
    ms = sorted(months)
    return ms[0] if len(ms) == 1 else f"{ms[0]} to {ms[-1]}"
