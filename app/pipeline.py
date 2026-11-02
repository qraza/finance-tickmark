"""End-to-end: question -> route -> semantic query and/or retrieval -> cited answer.

Week 2 (this file): the metric path is real. Claude plans a semantic query, MetricFlow
runs it, and the answer carries the numbers, the SQL and deterministic text.
Document and "why" questions use the BM25 baseline for citations; week 3 replaces
the router and adds an LLM-written explanation grounded in the retrieved passages.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from .answer import Answer, Citation
from .format import render
from .llm import plan_metric_query
from .semantic.query import SemanticQueryError


class Route(str, Enum):
    METRIC = "metric"      # "what was Q3 opex vs budget?"
    DOCUMENT = "document"  # "what is our FX policy?"
    WHY = "why"            # "why is Q3 opex over budget?" -> number + explanation


def route(question: str) -> Route:
    # TODO(week 3): replace with an LLM classifier and add routing cases to the golden set.
    q = question.lower()
    if q.startswith("why") or "explain" in q or "driver" in q:
        return Route.WHY
    if any(w in q for w in ("policy", "convention", "how do we", "definition", "good or bad", "have to explain")):
        return Route.DOCUMENT
    return Route.METRIC


@lru_cache(maxsize=1)
def default_engine():
    from .semantic.engine import SemanticEngine

    engine = SemanticEngine()
    engine.load_dimension_values()
    return engine


@lru_cache(maxsize=1)
def default_index():
    from .rag.index import BM25Index

    return BM25Index.from_corpus()


def answer_metric(question: str, *, client=None, engine=None) -> Answer:
    engine = engine or default_engine()
    plan = plan_metric_query(question, engine.catalogue, client=client)
    if isinstance(plan, str):
        return Answer.refusal(plan)
    try:
        res = engine.run(plan)
    except SemanticQueryError as e:
        return Answer.refusal(str(e))
    except Exception as e:  # noqa: BLE001 - a query that won't compile is refused, never guessed
        return Answer.refusal(f"the semantic layer could not run that query ({type(e).__name__})")

    metric_cols = [c for c in res.columns if c in engine.catalogue.metrics]
    if not res.rows or all(r[res.columns.index(c)] is None for r in res.rows for c in metric_cols):
        return Answer.refusal("there is no data for that period; actuals run to September 2026")

    first = dict(zip(res.columns, res.rows[0]))
    numbers = {c: float(first[c]) for c in metric_cols if first[c] is not None}
    return Answer(
        text=render(plan, res.columns, res.rows, engine.catalogue),
        numbers=numbers,
        sql=res.sql,
        semantic_query=plan.to_dict(),
    )


def answer_document(question: str, *, index=None, k: int = 3) -> Answer:
    hits = (index or default_index()).search(question, k=k)
    if not hits:
        return Answer.refusal("nothing in the documents matches that question")
    top = hits[0][0]
    excerpt = top.text.split("\n", 2)[-1].strip()
    return Answer(
        text=f"{excerpt}\n\n[{top.source} › {top.section}]",
        citations=[Citation(c.source, c.section) for c, _ in hits],
    )


def answer(question: str, *, client=None, engine=None, index=None) -> Answer:
    r = route(question)
    if r is Route.METRIC:
        return answer_metric(question, client=client, engine=engine)
    if r is Route.DOCUMENT:
        return answer_document(question, index=index)

    # WHY: the number from the semantic layer, the reason from the documents.
    num = answer_metric(question, client=client, engine=engine)
    doc = answer_document(question, index=index)
    if num.refused and doc.refused:
        return num
    text = "\n\n".join(t for t in [None if num.refused else num.text,
                                   None if doc.refused else "Context:\n" + doc.text] if t)
    return Answer(text=text, numbers=num.numbers, sql=num.sql,
                  citations=doc.citations, semantic_query=num.semantic_query)
