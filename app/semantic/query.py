"""A structured semantic query: the only thing the LLM produces for a number.

The LLM fills this in (via tool use, with metric and dimension names as enums).
It never writes SQL and never writes MetricFlow Jinja; ``to_mf_args`` renders the
filters, so a malformed or injected filter has nowhere to go.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalogue import TIME_GRAINS, Catalogue


class SemanticQueryError(ValueError):
    """The question asks for something the semantic layer cannot answer."""


@dataclass
class Filter:
    dimension: str
    values: list[str]          # one value means "=", several mean "IN"
    negate: bool = False


@dataclass
class TimeRange:
    grain: str                 # month | quarter | year
    start: str                 # ISO date of the first period, e.g. 2026-07-01
    end: str | None = None     # ISO date of the last period (inclusive); None = same as start


@dataclass
class SemanticQuery:
    metrics: list[str]
    group_by: list[str] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    time_range: TimeRange | None = None
    order_by: list[str] = field(default_factory=list)  # prefix "-" for descending
    limit: int | None = None

    # --- guardrails -------------------------------------------------------

    def validate(self, cat: Catalogue) -> None:
        if not self.metrics:
            raise SemanticQueryError("No metric requested.")
        unknown = [m for m in self.metrics if m not in cat.metrics]
        if unknown:
            raise SemanticQueryError(f"Not a governed metric: {', '.join(unknown)}.")
        allowed = set(cat.group_by_names)
        bad = [g for g in self.group_by if g not in allowed]
        bad += [f.dimension for f in self.filters if f.dimension not in cat.dimensions]
        if bad:
            raise SemanticQueryError(f"Unknown dimension: {', '.join(bad)}.")
        for o in self.order_by:
            if o.lstrip("-") not in set(self.metrics) | set(self.group_by):
                raise SemanticQueryError(f"Cannot order by {o}: not selected.")
        if self.time_range:
            if self.time_range.grain not in TIME_GRAINS:
                raise SemanticQueryError(f"Unsupported time grain: {self.time_range.grain}.")
            for d in filter(None, (self.time_range.start, self.time_range.end)):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                    raise SemanticQueryError(f"Bad date: {d}.")
        if self.limit is not None and not (0 < self.limit <= 1000):
            raise SemanticQueryError("Limit must be between 1 and 1000.")

    # --- rendering --------------------------------------------------------

    def where_clauses(self) -> list[str]:
        clauses = []
        for f in self.filters:
            vals = [_quote(v) for v in f.values]
            col = "{{ Dimension('" + f.dimension + "') }}"
            if len(vals) == 1:
                clauses.append(f"{col} {'!=' if f.negate else '='} {vals[0]}")
            else:
                clauses.append(f"{col} {'not in' if f.negate else 'in'} ({', '.join(vals)})")
        if self.time_range:
            t = "{{ TimeDimension('metric_time', '" + self.time_range.grain + "') }}"
            end = self.time_range.end or self.time_range.start
            if end == self.time_range.start:
                clauses.append(f"{t} = '{self.time_range.start}'")
            else:
                clauses.append(f"{t} >= '{self.time_range.start}' and {t} <= '{end}'")
        return clauses

    def to_mf_args(self) -> dict:
        return dict(
            metric_names=self.metrics,
            group_by_names=self.group_by or None,
            where_constraints=self.where_clauses() or None,
            order_by_names=self.order_by or None,
            limit=self.limit,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticQuery":
        return cls(
            metrics=list(d.get("metrics", [])),
            group_by=list(d.get("group_by", [])),
            filters=[Filter(**f) for f in d.get("filters", [])],
            time_range=TimeRange(**d["time_range"]) if d.get("time_range") else None,
            order_by=list(d.get("order_by", [])),
            limit=d.get("limit"),
        )


def _quote(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"
