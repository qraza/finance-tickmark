from app.answer import Answer, Citation
from app.rag.index import BM25Index
from evals.core import Case
from evals.finance import citation_recall, numeric_accuracy, refusal_correctness


def test_numeric_tolerance():
    case = Case("c", "q", expected={"numbers": {"opex": 1_000_000}})
    assert numeric_accuracy(case, Answer("", numbers={"opex": 1_004_000})) == 1.0
    assert numeric_accuracy(case, Answer("", numbers={"opex": 1_010_000})) == 0.0


def test_refusal_scored_both_ways():
    should = Case("c", "q", expected={"refuse": True})
    shouldnt = Case("c", "q", expected={"numbers": {"x": 1}})
    assert refusal_correctness(should, Answer.refusal("no")) == 1.0
    assert refusal_correctness(should, Answer("made it up", numbers={"x": 1})) == 0.0
    assert refusal_correctness(shouldnt, Answer.refusal("no")) == 0.0


def test_citation_recall():
    case = Case("c", "q", expected={"sources": ["a.md", "b.md"]})
    assert citation_recall(case, Answer("", citations=[Citation("a.md")])) == 0.5


def test_bm25_finds_engineering_note():
    hits = BM25Index.from_corpus().search("contractor ramp platform migration approval", k=3)
    assert "notes/engineering_q3_migration.md" in {c.source for c, _ in hits}
