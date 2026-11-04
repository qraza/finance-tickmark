"""Guardrails on the semantic query. Needs the manifest (`make build`), not the warehouse."""

import pytest

from app.llm import plan_tools
from app.semantic.catalogue import Catalogue
from app.semantic.query import SemanticQuery, SemanticQueryError


@pytest.fixture(scope="module")
def cat():
    return Catalogue.load()


def q(**kw):
    return SemanticQuery.from_dict(kw)


def test_catalogue_has_owners(cat):
    assert "opex_variance" in cat.metrics
    assert all(m.owner for m in cat.metrics.values())


def test_unknown_metric_refused(cat):
    with pytest.raises(SemanticQueryError, match="governed metric"):
        q(metrics=["net_revenue_retention"]).validate(cat)


def test_unknown_dimension_refused(cat):
    with pytest.raises(SemanticQueryError, match="Unknown dimension"):
        q(metrics=["opex"], group_by=["customer__name"]).validate(cat)


def test_bad_date_refused(cat):
    with pytest.raises(SemanticQueryError):
        q(metrics=["opex"], time_range={"grain": "quarter", "start": "2026-07-01'; drop table x;--"}).validate(cat)


def test_filter_values_are_escaped(cat):
    sq = q(metrics=["opex"], filters=[{"dimension": "cost_centre__cost_centre_name", "values": ["O'Brien"]}])
    sq.validate(cat)
    assert "'O''Brien'" in sq.where_clauses()[0]


def test_tool_schema_enumerates_metrics(cat):
    schema = plan_tools(cat)[0]["input_schema"]
    assert set(schema["properties"]["metrics"]["items"]["enum"]) == set(cat.metrics)
