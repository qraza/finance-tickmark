"""Check that a plan covers the cost centres, accounts and entities the question names.

The fourth pipeline run caught Claude answering "What drove the Customer Operations
underspend in August?" with company-wide August figures: it dropped the filter. This
module finds dimension values named in the question and reports any that the plan
neither filters on nor groups by. The pipeline re-plans once with the gap spelt out,
then adds the filter in code if it is still missing.

Matching is exact, to avoid false alarms: single words must match case ("Engineering",
not "engineering costs"), multi-word values match in any case ("customer operations").
"""

from __future__ import annotations

import re

from .semantic.catalogue import Catalogue
from .semantic.query import Filter, SemanticQuery

# Values too generic to be a mention: "opex" is in most questions, "True" in none.
SKIP_DIMENSIONS = {"account__account_type", "pnl_line__is_closed_period", "legal_entity__currency"}


def named_values(question: str, cat: Catalogue) -> list[tuple[str, str]]:
    """(dimension, value) pairs named in the question, longest values first."""
    found = []
    for dim, d in cat.dimensions.items():
        if dim in SKIP_DIMENSIONS:
            continue
        for v in d.values:
            flags = re.IGNORECASE if " " in v else 0
            if len(v) >= 3 and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", question, flags):
                found.append((dim, v))
    # "Marketing programmes" also contains "Marketing": keep the longer, more specific one
    found.sort(key=lambda dv: -len(dv[1]))
    kept: list[tuple[str, str]] = []
    for dim, v in found:
        if not any(v.lower() in k.lower() and v != k for _, k in kept):
            kept.append((dim, v))
    return kept


def uncovered(question: str, query: SemanticQuery, cat: Catalogue) -> list[tuple[str, str]]:
    """Named values the plan neither filters on nor groups by."""
    def covered(dim: str, v: str) -> bool:
        if dim in query.group_by:
            return True
        return any(f.dimension == dim and not f.negate and v in f.values for f in query.filters)

    return [(dim, v) for dim, v in named_values(question, cat) if not covered(dim, v)]


def describe(missing: list[tuple[str, str]]) -> str:
    items = ", ".join(f"{v} ({dim})" for dim, v in missing)
    return (f"The question names {items}, but your query neither filters on nor groups by "
            f"{'it' if len(missing) == 1 else 'them'}. Add the filter if the question is about it.")


def add_filters(query: SemanticQuery, missing: list[tuple[str, str]]) -> None:
    for dim, v in missing:
        existing = next((f for f in query.filters if f.dimension == dim and not f.negate), None)
        if existing:
            existing.values = [v]  # the question names one value; narrow to it
        else:
            query.filters.append(Filter(dimension=dim, values=[v]))
