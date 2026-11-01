"""End-to-end: question -> route -> semantic query and/or retrieval -> cited answer.

Build order (see README):
  week 2  metric path: plan_metric_query -> SemanticEngine -> Answer with numbers + SQL
  week 3  document path (BM25 baseline exists) and the "why" path that combines both
"""

from __future__ import annotations

from enum import Enum

from .answer import Answer


class Route(str, Enum):
    METRIC = "metric"      # "what was Q3 opex vs budget?"
    DOCUMENT = "document"  # "what is our FX policy?"
    WHY = "why"            # "why is Q3 opex over budget?" -> number + explanation


def route(question: str) -> Route:
    # TODO(week 3): replace with an LLM classifier and add routing cases to the golden set.
    q = question.lower()
    if q.startswith("why") or "explain" in q or "driver" in q:
        return Route.WHY
    if any(w in q for w in ("policy", "convention", "how do we", "definition")):
        return Route.DOCUMENT
    return Route.METRIC


def answer(question: str) -> Answer:
    raise NotImplementedError("Wire the metric path in week 2; see app/llm.py and app/semantic/engine.py.")
