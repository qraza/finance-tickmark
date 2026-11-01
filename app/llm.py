"""Question -> SemanticQuery, via tool use.

The tool schema is generated from the catalogue, with metric and dimension names as
JSON-schema enums. The model can pick a governed metric or call ``refuse``; there is
no free-text field it could smuggle a made-up metric or raw SQL through.

Week 2 of the build: wire ``plan_metric_query`` into ``app.pipeline`` and grow the
golden set. Needs ANTHROPIC_API_KEY; unit tests only exercise ``tool_schemas``.
"""

from __future__ import annotations

import os

from .semantic.catalogue import TIME_GRAINS, Catalogue
from .semantic.query import SemanticQuery

MODEL = os.environ.get("TICKMARK_MODEL", "claude-sonnet-5-5")

SYSTEM = """You translate finance questions about Larkspur Ltd into queries against a governed semantic layer.
Rules:
- Use only the metrics and dimensions in the tool schema. If the question needs a metric that does not exist,
  call `refuse` and say which metric is missing. Never approximate one metric with another.
- Variances use the *_variance metrics, which compare closed months only. Actuals exist from January to September 2026.
  If the question asks for actuals in a month that is not closed, call `refuse`.
- Fiscal year = calendar year. Q3 2026 starts 2026-07-01.

{catalogue}"""


def tool_schemas(cat: Catalogue) -> list[dict]:
    dims = sorted(cat.dimensions)
    return [
        {
            "name": "query_metrics",
            "description": "Query governed metrics from the semantic layer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "minItems": 1,
                                "items": {"type": "string", "enum": sorted(cat.metrics)}},
                    "group_by": {"type": "array", "items": {"type": "string", "enum": cat.group_by_names}},
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "dimension": {"type": "string", "enum": dims},
                                "values": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "negate": {"type": "boolean"},
                            },
                            "required": ["dimension", "values"],
                        },
                    },
                    "time_range": {
                        "type": "object",
                        "properties": {
                            "grain": {"type": "string", "enum": list(TIME_GRAINS)},
                            "start": {"type": "string", "description": "ISO date of first period, e.g. 2026-07-01"},
                            "end": {"type": "string", "description": "ISO date of last period, inclusive"},
                        },
                        "required": ["grain", "start"],
                    },
                    "order_by": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                "required": ["metrics"],
            },
        },
        {
            "name": "refuse",
            "description": "Use when the semantic layer cannot answer the question.",
            "input_schema": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    ]


def plan_metric_query(question: str, cat: Catalogue, client=None) -> SemanticQuery | str:
    """Return a SemanticQuery, or a refusal reason string."""
    if client is None:
        import anthropic  # optional dependency: pip install -e ".[llm]"

        client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM.format(catalogue=cat.describe()),
        tools=tool_schemas(cat),
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": question}],
    )
    call = next(b for b in resp.content if b.type == "tool_use")
    if call.name == "refuse":
        return call.input["reason"]
    return SemanticQuery.from_dict(call.input)
