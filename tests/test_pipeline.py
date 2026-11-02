"""The metric path end to end, with a fake Claude. Needs `make build` (DuckDB + manifest), no API key.

The fake returns a fixed tool call, so these tests check everything *around* the LLM:
validation, MetricFlow, refusal on empty data, and that numbers in the text match the query.
The LLM's own choices are scored by `make evals-pipeline` with a real key.
"""

from types import SimpleNamespace

import pytest

from app.format import money, value
from app.pipeline import Route, answer, route


class FakeClaude:
    def __init__(self, name: str, tool_input: dict):
        self.block = SimpleNamespace(type="tool_use", name=name, input=tool_input)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[self.block])


Q3 = {"grain": "quarter", "start": "2026-07-01"}


@pytest.fixture(scope="module")
def engine():
    from app.pipeline import default_engine

    return default_engine()


def test_metric_answer_numbers_and_sql(engine):
    fake = FakeClaude("query_metrics", {"metrics": ["opex", "opex_budget_to_date", "opex_variance"], "time_range": Q3})
    ans = answer("What was Q3 opex against budget?", client=fake, engine=engine)
    assert not ans.refused
    assert ans.numbers["opex_variance"] == pytest.approx(304948.87, abs=0.01)
    assert "£305k adverse" in ans.text and "Q3 2026" in ans.text
    assert ans.sql and "fct_pnl" in ans.sql


def test_prompt_carries_catalogue_and_values(engine):
    fake = FakeClaude("query_metrics", {"metrics": ["opex"], "time_range": Q3})
    answer("Q3 opex?", client=fake, engine=engine)
    call = fake.calls[0]
    assert "Engineering" in call["system"] and "opex_variance" in call["system"]
    assert call["tool_choice"] == {"type": "auto"}  # 5.5 models reject forced tool use
    assert all(tool["strict"] for tool in call["tools"])


def test_text_only_reply_is_refused(engine):
    class TextOnly(FakeClaude):
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Opex was about £5m.")])

    ans = answer("Q3 opex?", client=TextOnly("x", {}), engine=engine)
    assert ans.refused and not ans.numbers


def test_model_refusal_passes_through(engine):
    fake = FakeClaude("refuse", {"reason": "no net revenue retention metric"})
    ans = answer("What was NRR in Q3?", client=fake, engine=engine)
    assert ans.refused and "retention" in ans.refusal_reason


def test_unclosed_month_refused_even_if_model_queries_it(engine):
    fake = FakeClaude("query_metrics", {"metrics": ["opex"], "time_range": {"grain": "month", "start": "2026-10-01"}})
    ans = answer("Opex actuals in October 2026?", client=fake, engine=engine)
    assert ans.refused and "September" in ans.refusal_reason


def test_invented_dimension_value_refused(engine):
    fake = FakeClaude("query_metrics", {
        "metrics": ["opex"], "time_range": Q3,
        "filters": [{"dimension": "cost_centre__cost_centre_name", "values": ["Research"]}],
    })
    ans = answer("Q3 opex for Research?", client=fake, engine=engine)
    assert ans.refused and "Research" in ans.refusal_reason


def test_malformed_tool_input_refused(engine):
    fake = FakeClaude("query_metrics", {"metrics": ["opex"], "filters": [{"dim": "x"}]})
    assert answer("Q3 opex?", client=fake, engine=engine).refused


def test_ranking_uses_first_row(engine):
    fake = FakeClaude("query_metrics", {
        "metrics": ["opex_variance"], "group_by": ["cost_centre__cost_centre_name"],
        "time_range": Q3, "order_by": ["-opex_variance"], "limit": 1,
    })
    ans = answer("Which cost centre overspent most in Q3?", client=fake, engine=engine)
    assert "Engineering" in ans.text
    assert ans.numbers["opex_variance"] == pytest.approx(248443.45, abs=0.01)


def test_why_combines_number_and_citations(engine):
    fake = FakeClaude("query_metrics", {"metrics": ["opex_variance"], "time_range": Q3})
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert ans.numbers["opex_variance"] == pytest.approx(304948.87, abs=0.01)
    assert ans.citations


def test_routes():
    assert route("Why is opex over?") is Route.WHY
    assert route("Is a positive opex variance good or bad?") is Route.DOCUMENT
    assert route("What was Q3 revenue?") is Route.METRIC


def test_formatting():
    assert money(5_633_428.87) == "£5.63m"
    assert money(-56_311.65) == "-£56k"
    assert value("revenue_variance", -130_728.77) == "-£131k adverse"
    assert value("opex_variance", -56_311.65) == "-£56k favourable"
    assert value("ebitda_margin", 0.256) == "25.6%"
