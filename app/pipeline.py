"""End-to-end: question -> Claude's plan -> semantic layer and/or documents -> cited answer.

    metric question         plan -> period check -> MetricFlow -> figures (formatted in code)
    "why" question          plan (explain=true) -> period check -> MetricFlow -> drill-down by
                            cost centre and account -> retrieval scoped by the period and
                            targeted at each driver -> Claude writes the reason -> grounding check
    document question       plan -> retrieval -> Claude writes the answer -> grounding check
    anything else           refused

Claude routes by choosing a tool, so there is no keyword router. If the written answer
fails the grounding check, cites nothing, or doesn't come back, the answer falls back
to the figures and the best-matching passage, with a warning, rather than guessing.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

from .answer import Answer, Citation
from .format import render
from .grounding import ungrounded
from .format import value
from .llm import plan, write_answer
from .periods import asked_range, mismatch
from .semantic.query import SemanticQuery, SemanticQueryError, TimeRange

_engine_lock = threading.Lock()
_engine = None


def default_engine():
    """One shared engine; thread-safe so evals can run questions in parallel."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from .semantic.engine import SemanticEngine

            engine = SemanticEngine()
            engine.load_dimension_values()
            _engine = engine
        return _engine


@lru_cache(maxsize=1)
def default_index():
    from .rag.index import BM25Index

    return BM25Index.from_corpus()


# --- retrieval -------------------------------------------------------------------

def retrieve(question: str, semantic_query: dict | None = None, *, index=None, k: int = 5):
    """Search the corpus for a question, scoped by the semantic query when there is one.

    The period limits monthly commentary to the months asked about; filter values
    (e.g. "Marketing") are added to the search terms. Undated documents always count.
    """
    from .rag.scope import scope

    periods, terms = scope(semantic_query)
    return (index or default_index()).search(" ".join([question, *terms]), k=k, periods=periods)


DRILL_DIMS = ["cost_centre__cost_centre_name", "account__account_name", "account__account_type"]
WHY_EXCLUDE = ("metrics/",)  # metric definitions say what a metric is, never why it moved


def drivers(query: SemanticQuery, engine, n: int = 6) -> list[dict]:
    """The largest contributions to the asked metric, by cost centre and account.

    Same metric, filters and period as the question; grouped so each row says which
    cost centre and account moved, and what type the account is (revenue, cogs, opex).
    """
    metric = next((m for m in query.metrics if m.endswith("_variance")), query.metrics[0])
    if metric.endswith(("_pct", "_margin")):
        return []
    drill = SemanticQuery(metrics=[metric], group_by=DRILL_DIMS, filters=query.filters,
                          time_range=query.time_range)
    try:
        res = engine.run(drill)
    except Exception:  # noqa: BLE001 - the drill-down is extra context, never required
        return []
    rows = [dict(zip(res.columns, r)) for r in res.rows if r[res.columns.index(metric)] is not None]
    total = sum(abs(r[metric]) for r in rows) or 1.0
    rows = sorted(rows, key=lambda r: -abs(r[metric]))
    out = [
        {"cost_centre": r["cost_centre__cost_centre_name"], "account": r["account__account_name"],
         "type": r["account__account_type"], "metric": metric, "value": float(r[metric])}
        for r in rows[:n] if abs(r[metric]) >= 0.05 * total
    ]
    if metric.startswith("revenue"):
        fx = _fx_impact(query, engine)
        if fx is not None and abs(fx) >= 0.05 * total:
            out.append({"cost_centre": None, "account": "FX translation", "type": "revenue",
                        "metric": "fx_impact_on_revenue", "value": fx})
    return out


def _fx_impact(query: SemanticQuery, engine) -> float | None:
    """How much of a revenue variance is currency: actual vs budget exchange rates."""
    try:
        res = engine.run(SemanticQuery(metrics=["fx_impact_on_revenue"], filters=query.filters,
                                       time_range=query.time_range))
        v = res.rows[0][0] if res.rows else None
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def drivers_text(ds: list[dict], engine) -> str:
    if not ds:
        return ""
    label = engine.catalogue.metrics[ds[0]["metric"]].label
    lines = [f"- {d['cost_centre']} / {d['account']} ({d['type']}): {value(d['metric'], d['value'])}"
             for d in ds if d["cost_centre"]]
    text = f"Largest drivers ({label}, by cost centre and account):\n" + "\n".join(lines)
    fx = next((d for d in ds if d["account"] == "FX translation"), None)
    if fx:
        text += (f"\nOf which FX translation: {value(fx['metric'], fx['value'])} "
                 "(actuals at actual exchange rates vs the budget rate)")
    return text


def why_passages(question: str, semantic_query: dict | None, ds: list[dict], *, index=None, cap: int = 8):
    """Passages for a "why" question: one scoped search on the question, plus one per driver.

    The per-driver searches make sure each material driver (Engineering contractors,
    the Marketing summit) gets its own explanation, not only the biggest one. Passages
    that repeat one already picked (April and May say the same thing) are skipped, so
    each slot brings something new.
    """
    from .rag.scope import scope

    idx = index or default_index()
    periods, terms = scope(semantic_query)
    searches = [(" ".join([question, *terms]), 3)]
    for d in ds[:3]:
        if d["account"] == "FX translation":
            searches.append(("currency dollar weakened exchange rate budget rate constant currency", 2))
        else:
            searches.append((f"{d['cost_centre']} {d['account']} variance budget", 2))
    picked, seen = [], set()
    for text, n in searches:
        new = 0
        for c, _ in idx.search(text, k=n + 4, periods=periods, exclude=WHY_EXCLUDE):
            key = _gist(c.body)
            if c.id in {p.id for p in picked} or key in seen:
                continue
            picked.append(c)
            seen.add(key)
            new += 1
            if new == n:
                break
    return picked[:cap]


def _gist(text: str) -> str:
    """Text with numbers removed: two months' notes that differ only in figures match."""
    return re.sub(r"[£$%\d.,-]+", "#", text.lower())


# --- the three paths -------------------------------------------------------------

def run_metric(query: SemanticQuery, engine) -> Answer:
    try:
        res = engine.run(query)
    except SemanticQueryError as e:
        return Answer.refusal(str(e))
    except Exception as e:  # noqa: BLE001 - a query that won't compile is refused, never guessed
        return Answer.refusal(f"the semantic layer could not run that query ({type(e).__name__})")

    metric_cols = [c for c in res.columns if c in engine.catalogue.metrics]
    if not res.rows or all(r[res.columns.index(c)] is None for r in res.rows for c in metric_cols):
        return Answer.refusal("there is no data for that period; actuals run to September 2026")

    first = dict(zip(res.columns, res.rows[0]))
    return Answer(
        text=render(query, res.columns, res.rows, engine.catalogue),
        numbers={c: float(first[c]) for c in metric_cols if first[c] is not None},
        sql=res.sql,
        semantic_query=query.to_dict(),
    )


def explain(question: str, num: Answer, query: SemanticQuery, *, engine, client=None, index=None) -> Answer:
    """A metric answer plus the reason, written by Claude from the drill-down and passages."""
    ds = drivers(query, engine)
    figures = "\n\n".join(filter(None, [num.text, drivers_text(ds, engine)]))
    hits = why_passages(question, num.semantic_query, ds, index=index)
    if not hits:
        return Answer(text=figures, numbers=num.numbers, sql=num.sql, semantic_query=num.semantic_query,
                      warnings=["no documents matched; showing the figures only"])
    written = write_answer(question, figures, hits, client=client)
    if written is not None and not written.answerable:
        return Answer(
            text=f"{figures}\n\nThe documents don't explain this: {written.text}",
            numbers=num.numbers, sql=num.sql, semantic_query=num.semantic_query,
            warnings=["the documents don't explain this variance"],
        )
    raw = list(num.numbers.values()) + [d["value"] for d in ds]
    body, citations, warning = _accept(written, hits, [figures], raw)
    if body is None:
        body, citations = _fallback_text(hits), _top_citations(hits)
    return Answer(
        text=f"{figures}\n\n{body}",
        numbers=num.numbers, sql=num.sql, semantic_query=num.semantic_query,
        citations=citations, warnings=[w for w in [warning] if w],
    )


def answer_document(question: str, search: str | None = None, *, client=None, index=None) -> Answer:
    hits = [c for c, _ in retrieve(" ".join(filter(None, [question, search])), index=index)]
    if not hits:
        return Answer.refusal("nothing in the documents matches that question")
    written = write_answer(question, None, hits, client=client)
    if written is not None and not written.answerable:
        return Answer.refusal(f"the documents don't answer that ({written.text})")
    body, citations, warning = _accept(written, hits, [], [])
    if body is None:
        body, citations = _fallback_text(hits), _top_citations(hits)
    return Answer(text=body, citations=citations, warnings=[w for w in [warning] if w])


def answer(question: str, *, client=None, engine=None, index=None) -> Answer:
    engine = engine or default_engine()
    p = plan(question, engine.catalogue, client=client)
    if p.kind == "refuse":
        return Answer.refusal(p.reason)
    if p.kind == "document":
        return answer_document(question, p.search, client=client, index=index)
    _fill_period(question, p.query)
    issue = mismatch(question, p.query.to_dict().get("time_range"))
    if issue:  # the plan covers a different period from the question: re-plan once, spelling it out
        first = p.query.to_dict()
        p = plan(question, engine.catalogue, client=client, feedback=issue)
        if p.kind == "refuse":
            return Answer.refusal(p.reason)
        if p.kind == "metric":
            _fill_period(question, p.query)
        if p.kind != "metric" or mismatch(question, p.query.to_dict().get("time_range")):
            ans = Answer.refusal("the query's period didn't match the period in the question")
            ans.semantic_query = p.query.to_dict() if p.kind == "metric" else first
            return ans
    num = run_metric(p.query, engine)
    if num.refused:
        num.semantic_query = p.query.to_dict()  # keep the plan, so evals show why it was refused
    if num.refused or not p.explain:
        return num
    return explain(question, num, p.query, engine=engine, client=client, index=index)


def _fill_period(question: str, query: SemanticQuery) -> None:
    """If the plan has no time range but the question names one period, use that period.

    Claude left the time range out of ranking queries ("which cost centre ... in Q3"),
    which silently means "all closed months". Filling it from the question is deterministic.
    """
    if query.time_range is None:
        r = asked_range(question)
        if r:
            query.time_range = TimeRange(**r)


# --- checks on what Claude wrote -------------------------------------------------

def _accept(written, hits, figure_texts: list[str], raw_numbers: list[float]):
    """(text, citations, warning). text None = rejected, use the fallback."""
    if written is None:
        return None, [], "the explanation could not be written; showing the best-matching passage"
    by_id = {c.id: c for c in hits}
    cited = [by_id[i] for i in dict.fromkeys(written.citation_ids) if i in by_id]
    if not cited:
        return None, [], "the explanation cited no passage; showing the best-matching passage"
    bad = ungrounded(written.text, figure_texts + [c.body for c in cited], raw_numbers)
    if bad:
        return None, [], f"the explanation stated figures not in its sources ({', '.join(bad)}); showing the passage"
    refs = "; ".join(f"{c.source} › {c.section}" for c in cited)
    return f"{written.text}\n[{refs}]", [Citation(c.source, c.section, c.body) for c in cited], None


def _fallback_text(hits) -> str:
    top = hits[0]
    return f"Relevant passage:\n{top.body}\n[{top.source} › {top.section}]"


def _top_citations(hits, n: int = 3) -> list[Citation]:
    return [Citation(c.source, c.section, c.body) for c in hits[:n]]
