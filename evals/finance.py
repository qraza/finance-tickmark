"""Finance-specific scorers and systems for the generic eval core."""

from __future__ import annotations

from functools import lru_cache

from app.answer import Answer, Citation
from evals.core import Case

REL_TOL = 0.005  # 0.5%: tolerates rounding in prose, not a wrong number


# --- scorers -----------------------------------------------------------------

def numeric_accuracy(case: Case, ans: Answer) -> float | None:
    """Share of expected numbers the answer got within tolerance."""
    expected = case.expected.get("numbers")
    if not expected or case.expected.get("refuse"):
        return None
    hits = 0
    for name, want in expected.items():
        got = ans.numbers.get(name)
        if got is not None and abs(got - want) <= REL_TOL * max(abs(want), 1.0):
            hits += 1
    return hits / len(expected)


def citation_recall(case: Case, ans: Answer) -> float | None:
    """`sources`: share of required sources cited. `sources_any`: 1 if any valid source is cited.

    Use `sources_any` when several documents explain the same thing equally well
    (the July commentary and the engineering note both explain the contractor overspend).
    """
    cited = {c.source for c in ans.citations}
    if case.expected.get("sources_any"):
        return float(bool(cited & set(case.expected["sources_any"])))
    expected = case.expected.get("sources")
    if not expected:
        return None
    return sum(s in cited for s in expected) / len(expected)


def citation_precision(case: Case, ans: Answer) -> float | None:
    """Share of cited sources that are valid for the question. Catches citing the wrong month.

    `sources_ok` lists documents that support the answer without being required (the budget
    memo, which explains why the summit was budgeted in June). Citing them isn't a mistake.

    Reported, not gated: the oracle cites all the passages it retrieved, while the pipeline cites
    only the passages Claude used, so only the pipeline's precision is meaningful.
    """
    valid = {s for key in ("sources", "sources_any", "sources_ok") for s in case.expected.get(key) or []}
    if not valid or not ans.citations:
        return None
    cited = {c.source for c in ans.citations}
    return len(cited & valid) / len(cited)


def refusal_correctness(case: Case, ans: Answer) -> float | None:
    """1 if the system refused exactly when it should have. Scored on every case."""
    return float(bool(case.expected.get("refuse")) == ans.refused)


SCORERS = {
    "numeric_accuracy": numeric_accuracy,
    "citation_recall": citation_recall,
    "citation_precision": citation_precision,
    "refusal_correctness": refusal_correctness,
}


# --- systems -----------------------------------------------------------------

def _engine():
    from app.pipeline import default_engine  # shared and thread-safe, so --workers works

    return default_engine()


@lru_cache(maxsize=1)
def _index():
    from app.rag.index import BM25Index

    return BM25Index.from_corpus()


def oracle(case: Case) -> Answer:
    """Runs each case's reference query and retrieval, with no LLM.

    This tests the substrate: if the oracle fails, the data, the semantic layer or
    the retriever is wrong, and no prompt can fix it. CI runs it on every push.
    Expected values in the golden set must come from the generator's ground truth
    (corpus/stories.yaml), never from the system under test.
    """
    from app.semantic.query import SemanticQuery, SemanticQueryError

    o = case.oracle or {}
    if o.get("refuse"):
        return Answer.refusal(o["refuse"])

    numbers, sql, citations = {}, None, []
    if "query" in o:
        q = SemanticQuery.from_dict(o["query"])
        try:
            res = _engine().run(q)
        except SemanticQueryError as e:
            return Answer.refusal(str(e))
        sql = res.sql
        row = res.rows[0] if res.rows else ()
        numbers = {c: float(v) for c, v in zip(res.columns, row) if isinstance(v, (int, float))}
    if o.get("retrieve"):
        # The pipeline's own retrieval on the raw question, as a perfect planner would see it:
        # "why" cases use the drill-down and driver-targeted search, others the scoped search.
        from app.pipeline import drivers, retrieve, why_passages

        if "why" in case.tags and "query" in o:
            ds = drivers(SemanticQuery.from_dict(o["query"]), _engine())
            chunks = why_passages(case.question, o["query"], ds, index=_index())
        else:
            chunks = [c for c, _ in retrieve(case.question, o.get("query"), index=_index())]
        citations = [Citation(c.source, c.section) for c in chunks]
    return Answer(text="(oracle)", numbers=numbers, sql=sql, citations=citations)


def pipeline(case: Case) -> Answer:
    """The real system: question in, answer out. Week 2 onwards."""
    from app.pipeline import answer

    return answer(case.question)
