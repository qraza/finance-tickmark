"""Turn a query result into answer text, deterministically.

No LLM writes the numbers in an answer: they are formatted here from the result
rows, so a figure in the text can't drift from the figure the query returned.
"""

from __future__ import annotations

from datetime import date

from .semantic.catalogue import Catalogue
from .semantic.query import SemanticQuery

# For these, a positive variance is good news (FP&A policy sign convention).
FAVOURABLE_WHEN_POSITIVE = ("revenue", "gross_profit", "ebitda")


def money(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}£{v / 1_000_000:,.2f}m"
    if v >= 1_000:
        return f"{sign}£{v / 1_000:,.0f}k"
    return f"{sign}£{v:,.0f}"


def value(metric: str, v: float) -> str:
    if metric.endswith("_pct") or metric.endswith("_margin"):
        return f"{v * 100:.1f}%"
    text = money(v)
    if metric.endswith("_variance") and v != 0:
        good = metric.startswith(FAVOURABLE_WHEN_POSITIVE) == (v > 0)
        text += " favourable" if good else " adverse"
    return text


def period(q: SemanticQuery) -> str | None:
    t = q.time_range
    if t is None:
        return None
    return label(t.grain, t.start) + ("" if not t.end or t.end == t.start else f" to {label(t.grain, t.end)}")


def label(grain: str, iso: str) -> str:
    d = date.fromisoformat(iso)
    if grain == "quarter":
        return f"Q{(d.month - 1) // 3 + 1} {d.year}"
    if grain == "year":
        return str(d.year)
    return d.strftime("%b %Y")


def scope(q: SemanticQuery) -> str:
    parts = [p for p in [period(q)] if p]
    for f in q.filters:
        dim = f.dimension.split("__", 1)[-1].replace("_", " ")
        op = "not " if f.negate else ""
        parts.append(f"{dim} {op}{' or '.join(f.values)}")
    return ", ".join(parts) or "All periods"


def render(q: SemanticQuery, columns: list[str], rows: list[tuple], cat: Catalogue) -> str:
    metric_cols = [c for c in columns if c in cat.metrics]
    group_cols = [c for c in columns if c not in cat.metrics]
    lines = [f"{scope(q)}:"]
    for row in rows:
        rec = dict(zip(columns, row))
        prefix = ", ".join(str(rec[g]) for g in group_cols)
        vals = "; ".join(
            f"{cat.metrics[m].label} {value(m, rec[m])}" for m in metric_cols if rec[m] is not None
        )
        lines.append(f"- {prefix + ': ' if prefix else ''}{vals}")
    return "\n".join(lines)
