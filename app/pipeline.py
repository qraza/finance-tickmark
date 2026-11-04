"""End-to-end: question -> Claude's plan -> semantic layer and/or documents -> cited answer.

    metric question         plan -> MetricFlow -> figures (formatted in code, no LLM prose)
    "why" question          plan (explain=true) -> MetricFlow -> retrieval scoped by the query
                            -> Claude writes the reason from passages -> grounding check
    document question       plan -> retrieval -> Claude writes the answer -> grounding check
    anything else           refused

Claude routes by choosing a tool, so there is no keyword router. If the written answer
fails the grounding check, cites nothing, or doesn't come back, the answer falls back
to the figures and the best-matching passage, with a warning, rather than guessing.
"""

from __future__ import annotations

from functools import lru_cache

from .answer import Answer, Citation
from .format import render
from .grounding import ungrounded
from .llm import plan, write_answer
from .semantic.query import SemanticQuery, SemanticQueryError


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


# --- retrieval -------------------------------------------------------------------

def retrieve(question: str, semantic_query: dict | None = None, *, index=None, k: int = 5):
    """Search the corpus for a question, scoped by the semantic query when there is one.

    The period limits monthly commentary to the months asked about; filter values
    (e.g. "Marketing") are added to the search terms. Undated documents always count.
    """
    from .rag.scope import scope

    periods, terms = scope(semantic_query)
    return (index or default_index()).search(" ".join([question, *terms]), k=k, periods=periods)


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


def explain(question: str, num: Answer, *, client=None, index=None) -> Answer:
    """A metric answer plus the reason, written by Claude from scoped passages."""
    hits = [c for c, _ in retrieve(question, num.semantic_query, index=index)]
    if not hits:
        num.warnings.append("no documents matched; showing the figures only")
        return num
    written = write_answer(question, num.text, hits, client=client)
    if written is not None and not written.answerable:
        return Answer(
            text=f"{num.text}\n\nThe documents don't explain this: {written.text}",
            numbers=num.numbers, sql=num.sql, semantic_query=num.semantic_query,
            warnings=["the documents don't explain this variance"],
        )
    body, citations, warning = _accept(written, hits, [num.text], list(num.numbers.values()))
    if body is None:
        body, citations = _fallback_text(hits), _top_citations(hits)
    return Answer(
        text=f"{num.text}\n\n{body}",
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
    num = run_metric(p.query, engine)
    if num.refused or not p.explain:
        return num
    return explain(question, num, client=client, index=index)


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
