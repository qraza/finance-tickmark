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
    """Share of expected sources the answer cited."""
    expected = case.expected.get("sources")
    if not expected:
        return None
    cited = {c.source for c in ans.citations}
    return sum(s in cited for s in expected) / len(expected)


def refusal_correctness(case: Case, ans: Answer) -> float | None:
    """1 if the system refused exactly when it should have. Scored on every case."""
    return float(bool(case.expected.get("refuse")) == ans.refused)


SCORERS = {
    "numeric_accuracy": numeric_accuracy,
    "citation_recall": citation_recall,
    "refusal_correctness": refusal_correctness,
}


# --- systems -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _engine():
    from app.semantic.engine import SemanticEngine

    return SemanticEngine()


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
    if "retrieve" in o:
        for chunk, _ in _index().search(o["retrieve"], k=o.get("k", 3)):
            citations.append(Citation(chunk.source, chunk.section))
    return Answer(text="(oracle)", numbers=numbers, sql=sql, citations=citations)


def pipeline(case: Case) -> Answer:
    """The real system: question in, answer out. Week 2 onwards."""
    from app.pipeline import answer

    return answer(case.question)
