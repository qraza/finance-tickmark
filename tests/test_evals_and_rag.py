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


def test_short_sections_stay_whole():
    """A metric definition and its owner come back as one passage (the EBITDA bug)."""
    chunks = BM25Index.from_corpus().chunks
    ebitda = [c for c in chunks if c.section == "EBITDA (`ebitda`)"]
    assert len(ebitda) == 1 and "Revenue minus cost of sales" in ebitda[0].body and "Owner" in ebitda[0].body


def test_long_sections_split_by_paragraph():
    notes = [c for c in BM25Index.from_corpus().chunks if c.source == "commentary/2026-07.md" and c.section == "Variance notes"]
    assert len(notes) == 3  # engineering, US revenue, customer operations


def test_period_detection():
    from app.periods import asked_months, mismatch

    assert asked_months("What was opex in Q3?") == {"2026-07", "2026-08", "2026-09"}
    assert asked_months("What drove the underspend in August?") == {"2026-08"}
    assert asked_months("What may drive opex?") is None  # "may" the verb, not May the month
    assert asked_months("Compare Q2 and Q3 opex") is None  # two periods: not checked
    assert mismatch("Opex in Q3?", {"grain": "quarter", "start": "2026-07-01"}) is None
    assert mismatch("Opex in Q3?", {"grain": "month", "start": "2026-01-01", "end": "2026-09-01"})


def test_time_ranges_read_like_metricflow():
    from app.rag.scope import months_in
    from app.semantic.query import SemanticQuery

    q3 = {"2026-07", "2026-08", "2026-09"}
    assert months_in("quarter", "2026-07-01") == q3
    assert months_in("quarter", "2026-07-01", "2026-09-01") == q3  # end = a date in the last quarter
    assert months_in("quarter", "2026-08-01") == q3  # snapped to the quarter's start
    sq = SemanticQuery.from_dict({"metrics": ["opex"], "time_range": {"grain": "quarter", "start": "2026-08-15"}})
    assert sq.where_clauses() == ["{{ TimeDimension('metric_time', 'quarter') }} = '2026-07-01'"]
