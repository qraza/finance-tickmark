import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_gen"))
import generate as g  # noqa: E402


def test_deterministic():
    assert g.build_ledger()[:2] == g.build_ledger()[:2]


def test_no_actuals_after_close():
    _, actual, _ = g.build_ledger()
    assert max(int(r["period"][5:7]) for r in actual) == 9


def test_planted_stories_are_material():
    budget, actual, _ = g.build_ledger()
    q2, q3 = [4, 5, 6], [7, 8, 9]

    def var(months, **f):
        return g.agg(actual, "actual", months, **f) - g.agg(budget, "budget", months, **f)

    # S1: Engineering contractors well over budget in Q3
    assert var(q3, cost_centre_code="CC300", account_code="6100") > 200_000
    # S2: Marketing programmes under in Q2 (timing)
    assert var(q2, cost_centre_code="CC200", account_code="6400") < -150_000
    # S3: US revenue under in GBP...
    assert var(q3, entity_code="US01", account_type="revenue") < -100_000
    # S4: Customer Operations under in Q3
    assert var(q3, cost_centre_code="CC400") < -100_000


def test_us_revenue_on_plan_in_constant_currency():
    budget, actual, _ = g.build_ledger()
    rows = [r for r in actual if r["entity_code"] == "US01" and r["account_code"].startswith("4")
            and int(r["period"][5:7]) in (7, 8, 9)]
    cc = sum(r["amount_local"] for r in rows) * g.BUDGET_USD_GBP
    b = g.agg(budget, "budget", [7, 8, 9], entity_code="US01", account_type="revenue")
    assert abs(cc / b - 1) < 0.02
