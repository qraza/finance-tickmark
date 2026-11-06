"""The pipeline end to end, with a fake Claude. Needs `make build` (DuckDB + manifest), no API key.

The fake replays scripted tool calls, so these tests check everything *around* the LLM:
validation, MetricFlow, refusals, scoped retrieval, citation checks and the grounding
check on written answers. The LLM's own choices are scored by `make evals-pipeline`.
"""

from types import SimpleNamespace

import pytest

from app.format import money, value
from app.pipeline import answer


def tool(name: str, tool_input: dict):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input)


class FakeClaude:
    """Returns one scripted response per call, in order: plan first, then write_answer."""

    def __init__(self, *blocks):
        self.blocks = list(blocks)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = self.blocks[min(len(self.calls), len(self.blocks)) - 1]
        return SimpleNamespace(content=[block])


def metric(**q):
    q.setdefault("explain", False)
    return tool("query_metrics", q)


def written(text: str, citations: list[str], answerable: bool = True):
    return tool("write_answer", {"answerable": answerable, "text": text, "citations": citations})


Q3 = {"grain": "quarter", "start": "2026-07-01"}
ENG_NOTE = "notes/engineering_q3_migration.md#2"  # Cost impact: "two and a half times budget"


@pytest.fixture(scope="module")
def engine():
    from app.pipeline import default_engine

    return default_engine()


# --- metric path ----------------------------------------------------------------

def test_metric_answer_numbers_and_sql(engine):
    fake = FakeClaude(metric(metrics=["opex", "opex_budget_to_date", "opex_variance"], time_range=Q3))
    ans = answer("What was Q3 opex against budget?", client=fake, engine=engine)
    assert not ans.refused
    assert ans.numbers["opex_variance"] == pytest.approx(304948.87, abs=0.01)
    assert "£305k adverse" in ans.text and "Q3 2026" in ans.text
    assert ans.sql and "fct_pnl" in ans.sql
    assert len(fake.calls) == 1  # no explanation call for a plain metric question


def test_plan_prompt_carries_catalogue_and_strict_tools(engine):
    fake = FakeClaude(metric(metrics=["opex"], time_range=Q3))
    answer("Q3 opex?", client=fake, engine=engine)
    call = fake.calls[0]
    assert "Engineering" in call["system"] and "opex_variance" in call["system"]
    assert call["tool_choice"] == {"type": "auto"}  # 5.5 models reject forced tool use
    assert {t["name"] for t in call["tools"]} == {"query_metrics", "search_documents", "refuse"}
    assert all(t["strict"] for t in call["tools"])


def test_model_refusal_passes_through(engine):
    ans = answer("What was NRR in Q3?", client=FakeClaude(tool("refuse", {"reason": "no NRR metric"})), engine=engine)
    assert ans.refused and "NRR" in ans.refusal_reason


def test_text_only_reply_is_refused(engine):
    fake = FakeClaude(SimpleNamespace(type="text", text="Opex was about £5m."))
    ans = answer("Q3 opex?", client=fake, engine=engine)
    assert ans.refused and not ans.numbers


def test_unclosed_month_refused_even_if_model_queries_it(engine):
    fake = FakeClaude(metric(metrics=["opex"], time_range={"grain": "month", "start": "2026-10-01"}))
    ans = answer("Opex actuals in October 2026?", client=fake, engine=engine)
    assert ans.refused and "September" in ans.refusal_reason


def test_invented_dimension_value_refused(engine):
    fake = FakeClaude(metric(metrics=["opex"], time_range=Q3,
                             filters=[{"dimension": "cost_centre__cost_centre_name", "values": ["Research"]}]))
    ans = answer("Q3 opex for Research?", client=fake, engine=engine)
    assert ans.refused and "Research" in ans.refusal_reason


def test_malformed_tool_input_refused(engine):
    fake = FakeClaude(tool("query_metrics", {"metrics": ["opex"], "explain": False, "filters": [{"dim": "x"}]}))
    assert answer("Q3 opex?", client=fake, engine=engine).refused


def test_ranking_uses_first_row(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], group_by=["cost_centre__cost_centre_name"],
                             time_range=Q3, order_by=["-opex_variance"], limit=1))
    ans = answer("Which cost centre overspent most in Q3?", client=fake, engine=engine)
    assert "Engineering" in ans.text
    assert ans.numbers["opex_variance"] == pytest.approx(248443.45, abs=0.01)


# --- why path -------------------------------------------------------------------

def test_why_writes_grounded_explanation_with_citations(engine):
    fake = FakeClaude(
        metric(metrics=["opex_variance"], time_range=Q3, explain=True),
        written("The overspend is mostly Engineering contractors, running at about two and a half times "
                "budget because the platform migration was brought forward.", [ENG_NOTE]),
    )
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert ans.numbers["opex_variance"] == pytest.approx(304948.87, abs=0.01)
    assert "migration was brought forward" in ans.text
    assert [c.source for c in ans.citations] == ["notes/engineering_q3_migration.md"]
    assert not ans.warnings


def test_why_passages_scoped_to_the_question_period(engine):
    fake = FakeClaude(
        metric(metrics=["opex_variance"], time_range={"grain": "quarter", "start": "2026-04-01"},
               filters=[{"dimension": "cost_centre__cost_centre_name", "values": ["Marketing"]}], explain=True),
        written("x", [], answerable=False),
    )
    answer("Why was Marketing under budget in Q2?", client=fake, engine=engine)
    prompt = fake.calls[1]["messages"][0]["content"]
    assert "commentary/2026-06.md" in prompt
    assert "commentary/2026-07.md" not in prompt and "commentary/2026-01.md" not in prompt


def test_invented_figure_falls_back_to_passage(engine):
    fake = FakeClaude(
        metric(metrics=["opex_variance"], time_range=Q3, explain=True),
        written("Contractors cost £512k, 40% above plan.", [ENG_NOTE]),
    )
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert "£512k" not in ans.text and "Relevant passage" in ans.text
    assert ans.warnings and "£512k" in ans.warnings[0]
    assert ans.numbers["opex_variance"] == pytest.approx(304948.87, abs=0.01)  # the number still stands


def test_explanation_citing_nothing_falls_back(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], time_range=Q3, explain=True),
                      written("Because of contractors.", []))
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert "Relevant passage" in ans.text and ans.warnings


def test_unexplained_variance_cites_nothing(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], time_range=Q3, explain=True),
                      written("No passage gives a reason.", [], answerable=False))
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert "don't explain" in ans.text and not ans.citations and not ans.refused


def test_why_includes_drill_down_with_account_types(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], time_range=Q3, explain=True),
                      written("Mostly Engineering contractors, after the migration was brought forward.", [ENG_NOTE]))
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert "Largest drivers" in ans.text
    assert "Engineering / Contractors (opex): £264k adverse" in ans.text
    assert "Marketing / Marketing programmes (opex)" in ans.text
    assert "Cloud hosting" not in ans.text  # cogs, not part of opex
    figures = fake.calls[1]["messages"][0]["content"]
    assert "Engineering / Contractors (opex)" in figures  # Claude sees the drivers too


def test_why_passages_cover_each_driver_not_just_the_biggest(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], time_range=Q3, explain=True), written("x", [], answerable=False))
    answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    tool_ids = fake.calls[1]["tools"][0]["input_schema"]["properties"]["citations"]["items"]["enum"]
    assert any(i.startswith("commentary/2026-09") for i in tool_ids)  # the Marketing summit
    assert any(i.startswith("notes/engineering") for i in tool_ids)
    assert not any(i.startswith("metrics/") for i in tool_ids)  # definitions never explain a variance


def test_figure_from_drill_down_counts_as_grounded(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], time_range=Q3, explain=True),
                      written("Engineering contractors are £264k adverse because the migration was brought forward.",
                              [ENG_NOTE]))
    ans = answer("Why is Q3 opex over budget?", client=fake, engine=engine)
    assert not ans.warnings and "£264k adverse because" in ans.text


def test_wrong_period_is_replanned(engine):
    ytd = {"grain": "month", "start": "2026-01-01", "end": "2026-09-01"}
    rank = dict(metrics=["opex_variance"], group_by=["cost_centre__cost_centre_name"], order_by=["opex_variance"], limit=1)
    fake = FakeClaude(metric(time_range=ytd, **rank), metric(time_range=Q3, **rank))
    ans = answer("Which cost centre was furthest under its opex budget in Q3?", client=fake, engine=engine)
    assert len(fake.calls) == 2 and "Correction" in fake.calls[1]["messages"][0]["content"]
    assert "Customer Operations" in ans.text
    assert ans.numbers["opex_variance"] == pytest.approx(-56311.65, abs=0.01)


def test_wrong_period_twice_is_refused(engine):
    ytd = {"grain": "month", "start": "2026-01-01", "end": "2026-09-01"}
    fake = FakeClaude(metric(metrics=["opex_variance"], time_range=ytd))
    ans = answer("What was the opex variance in Q3?", client=fake, engine=engine)
    assert ans.refused and "period" in ans.refusal_reason


def test_quarter_end_written_as_last_month_is_not_a_mismatch(engine):
    """Regression: Claude wrote Q3 as start 2026-07-01, end 2026-09-01 and the period check refused it."""
    q3 = {"grain": "quarter", "start": "2026-07-01", "end": "2026-09-01"}
    fake = FakeClaude(metric(metrics=["opex_variance"], group_by=["cost_centre__cost_centre_name"],
                             order_by=["-opex_variance"], limit=1, time_range=q3))
    ans = answer("Which cost centre had the biggest opex overspend in Q3?", client=fake, engine=engine)
    assert not ans.refused and len(fake.calls) == 1
    assert ans.numbers["opex_variance"] == pytest.approx(248443.45, abs=0.01)


def test_revenue_why_shows_fx_translation(engine):
    us = [{"dimension": "legal_entity__entity_code", "values": ["US01"]}]
    fake = FakeClaude(metric(metrics=["revenue_variance"], filters=us, time_range=Q3, explain=True),
                      written("x", [], answerable=False))
    answer("Why is US revenue below budget in Q3?", client=fake, engine=engine)
    figures = fake.calls[1]["messages"][0]["content"]
    assert "Of which FX translation: -£162k" in figures
    assert "The shortfall is currency" in figures  # the US commentary passage is offered


def test_ranking_plan_without_time_range_gets_the_question_period(engine):
    """Regression: Claude's actual plan for this question had no time_range at all (= YTD)."""
    fake = FakeClaude(tool("query_metrics", {"metrics": ["opex_variance"], "explain": False,
                                             "group_by": ["cost_centre__cost_centre_name"],
                                             "order_by": ["opex_variance"], "limit": 1}))
    ans = answer("Which cost centre was furthest under its opex budget in Q3?", client=fake, engine=engine)
    assert not ans.refused and len(fake.calls) == 1  # filled in code, no re-plan needed
    assert "Customer Operations" in ans.text and "Q3 2026" in ans.text
    assert ans.numbers["opex_variance"] == pytest.approx(-56311.65, abs=0.01)


def test_time_range_is_required_in_the_schema(engine):
    fake = FakeClaude(metric(metrics=["opex"], time_range=Q3))
    answer("Q3 opex?", client=fake, engine=engine)
    schema = next(t for t in fake.calls[0]["tools"] if t["name"] == "query_metrics")["input_schema"]
    assert "time_range" in schema["required"]


def test_refused_answer_keeps_the_plan(engine):
    fake = FakeClaude(metric(metrics=["opex"], time_range={"grain": "month", "start": "2026-10-01"}))
    ans = answer("Opex actuals in October 2026?", client=fake, engine=engine)
    assert ans.refused and ans.semantic_query["time_range"]["start"] == "2026-10-01"


def test_dropped_cost_centre_filter_is_replanned_then_added(engine):
    """Regression: Claude answered a Customer Operations question with company-wide August figures."""
    aug = {"grain": "month", "start": "2026-08-01"}
    company_wide = metric(metrics=["cogs_variance", "opex_variance"], time_range=aug, explain=False)
    fake = FakeClaude(company_wide, company_wide)  # ignores the correction too
    ans = answer("How far under budget was Customer Operations in August?", client=fake, engine=engine)
    assert len(fake.calls) == 2 and "Customer Operations" in fake.calls[1]["messages"][0]["content"]
    assert ans.numbers["opex_variance"] == pytest.approx(-18913.61, abs=0.01)  # filter added in code
    assert ans.numbers["cogs_variance"] == pytest.approx(-25065.0, abs=0.01)


def test_named_cost_centre_covered_by_group_by_is_fine(engine):
    fake = FakeClaude(metric(metrics=["opex_variance"], group_by=["cost_centre__cost_centre_name"], time_range=Q3))
    answer("How did Engineering compare with Marketing on opex in Q3?", client=fake, engine=engine)
    assert len(fake.calls) == 1


def test_filters_and_group_by_are_required_in_the_schema(engine):
    fake = FakeClaude(metric(metrics=["opex"], time_range=Q3))
    answer("Q3 opex?", client=fake, engine=engine)
    schema = next(t for t in fake.calls[0]["tools"] if t["name"] == "query_metrics")["input_schema"]
    assert {"filters", "group_by", "time_range"} <= set(schema["required"])


# --- document path --------------------------------------------------------------

def test_document_question_answered_from_policy(engine):
    fake = FakeClaude(
        tool("search_documents", {"query": "sign convention positive variance costs"}),
        written("For costs, a positive variance is adverse: it means spend is above budget.",
                ["policy/fpa_reporting_policy.md#1"]),
    )
    ans = answer("Is a positive opex variance good or bad?", client=fake, engine=engine)
    assert not ans.refused and not ans.numbers and not ans.sql
    assert [c.source for c in ans.citations] == ["policy/fpa_reporting_policy.md"]


def test_document_not_answerable_is_refused(engine):
    fake = FakeClaude(tool("search_documents", {"query": "parental leave policy"}),
                      written("The documents don't cover leave.", [], answerable=False))
    assert answer("What is the parental leave policy?", client=fake, engine=engine).refused


def test_citations_limited_to_retrieved_passages(engine):
    fake = FakeClaude(tool("search_documents", {"query": "sign convention"}),
                      written("Positive cost variances are adverse.", ["made/up.md#0"]))
    ans = answer("Is a positive opex variance good or bad?", client=fake, engine=engine)
    assert all(c.source != "made/up.md" for c in ans.citations) and ans.warnings


# --- formatting -----------------------------------------------------------------

def test_formatting():
    assert money(5_633_428.87) == "£5.63m"
    assert money(-56_311.65) == "-£56k"
    assert value("revenue_variance", -130_728.77) == "£131k adverse"  # the label carries the sign
    assert value("opex_variance", -56_311.65) == "£56k favourable"
    assert value("fx_impact_on_revenue", -161_516.98) == "-£162k"  # not a variance: keeps its sign


def test_source_labels_say_the_period():
    from app.format import source_label

    assert source_label("commentary/2026-07.md") == "Monthly commentary · July 2026 (that month's figures only)"
    assert source_label("notes/engineering_q3_migration.md") == "Cost-centre owner note"
    assert value("ebitda_margin", 0.256) == "25.6%"
