"""Generate the synthetic ledger and the matching document corpus for Larkspur Ltd.

Larkspur is a fictional UK SaaS company with a US subsidiary. Everything here is
deterministic (fixed seed), so the eval golden set can pin exact numbers.

Outputs
-------
dbt/seeds/*.csv           dimension tables, FX rates, GL actuals, budget
corpus/commentary/*.md    monthly management commentary, written from the data
corpus/stories.yaml       ground truth for the planted variance stories (used by evals)

The planted stories are the point of the dataset: each one is a variance in the
numbers *and* an explanation in the documents, so a "why" question needs both.

  S1  Engineering contractors overspend in Q3 (platform migration brought forward)
  S2  Marketing programmes: London summit moved from June to September
  S3  USD weakened in Q3, so US revenue is under budget in GBP but on plan in USD
  S4  Customer Operations hiring freeze: support costs under budget in Q3
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "dbt" / "seeds"
CORPUS = ROOT / "corpus"

SEED = 2026
YEAR = 2026
BUDGET_MONTHS = range(1, 13)
ACTUAL_MONTHS = range(1, 10)  # books closed through September (Q3)
NOISE = 0.02

ENTITIES = [
    # code, name, currency
    ("UK01", "Larkspur Ltd", "GBP"),
    ("US01", "Larkspur Inc", "USD"),
]

COST_CENTRES = [
    # code, name, owner
    ("CC100", "Commercial", "Priya Nair"),
    ("CC200", "Marketing", "Tom Achebe"),
    ("CC300", "Engineering", "Hannah Kowalski"),
    ("CC400", "Customer Operations", "Luis Ortega"),
    ("CC500", "G&A", "Mei Chen"),
]

ACCOUNTS = [
    # code, name, account_type
    ("4000", "Subscription revenue", "revenue"),
    ("4100", "Professional services revenue", "revenue"),
    ("5000", "Cloud hosting", "cogs"),
    ("5100", "Support staff costs", "cogs"),
    ("6000", "Salaries and benefits", "opex"),
    ("6100", "Contractors", "opex"),
    ("6200", "Software subscriptions", "opex"),
    ("6300", "Travel and entertainment", "opex"),
    ("6400", "Marketing programmes", "opex"),
    ("6500", "Rent and facilities", "opex"),
]

# Monthly budget base in local currency, keyed by (entity, cost_centre, account).
BASE = {
    ("UK01", "CC100", "4000"): 1_200_000,
    ("UK01", "CC100", "4100"): 150_000,
    ("US01", "CC100", "4000"): 1_100_000,
    ("US01", "CC100", "4100"): 120_000,
    ("UK01", "CC300", "5000"): 180_000,
    ("US01", "CC300", "5000"): 140_000,
    ("UK01", "CC400", "5100"): 90_000,
    ("US01", "CC400", "5100"): 75_000,
    ("UK01", "CC100", "6000"): 220_000,
    ("UK01", "CC200", "6000"): 90_000,
    ("UK01", "CC300", "6000"): 380_000,
    ("UK01", "CC400", "6000"): 70_000,
    ("UK01", "CC500", "6000"): 140_000,
    ("US01", "CC100", "6000"): 160_000,
    ("US01", "CC200", "6000"): 45_000,
    ("US01", "CC300", "6000"): 150_000,
    ("US01", "CC400", "6000"): 55_000,
    ("US01", "CC500", "6000"): 50_000,
    ("UK01", "CC300", "6100"): 60_000,
    ("UK01", "CC500", "6100"): 15_000,
    ("UK01", "CC100", "6200"): 12_000,
    ("UK01", "CC200", "6200"): 18_000,
    ("UK01", "CC300", "6200"): 40_000,
    ("UK01", "CC400", "6200"): 10_000,
    ("UK01", "CC500", "6200"): 22_000,
    ("US01", "CC100", "6200"): 9_000,
    ("US01", "CC300", "6200"): 15_000,
    ("UK01", "CC100", "6300"): 25_000,
    ("UK01", "CC200", "6300"): 12_000,
    ("US01", "CC100", "6300"): 20_000,
    ("UK01", "CC200", "6400"): 110_000,
    ("US01", "CC200", "6400"): 70_000,
    ("UK01", "CC500", "6500"): 75_000,
    ("US01", "CC500", "6500"): 30_000,
}

REVENUE_GROWTH = 0.015  # budgeted month-on-month growth on revenue lines
BUDGET_USD_GBP = 0.79   # budget FX rate: 1 USD = 0.79 GBP for the whole year

# Actual monthly USD->GBP rates. USD weakens from July (story S3).
ACTUAL_USD_GBP = {1: 0.792, 2: 0.788, 3: 0.791, 4: 0.789, 5: 0.793, 6: 0.787,
                  7: 0.752, 8: 0.748, 9: 0.751}


@dataclass(frozen=True)
class Story:
    id: str
    title: str
    multipliers: dict  # (entity, cc, account, month) -> actual/budget multiplier


def _story_multipliers() -> list[Story]:
    s1 = {}
    for m, mult in {7: 2.3, 8: 2.6, 9: 2.5}.items():
        s1[("UK01", "CC300", "6100", m)] = mult
        s1[("UK01", "CC300", "5000", m)] = 1.15  # dual-running hosting during migration
    s2 = {}
    for ent in ("UK01", "US01"):
        for m in (4, 5, 6):
            s2[(ent, "CC200", "6400", m)] = 0.55
        s2[(ent, "CC200", "6400", 9)] = 1.9
    s4 = {}
    for ent in ("UK01", "US01"):
        for m in (7, 8, 9):
            s4[(ent, "CC400", "5100", m)] = 0.85
            s4[(ent, "CC400", "6000", m)] = 0.85
    return [
        Story("S1", "Engineering contractors overspend (platform migration brought forward)", s1),
        Story("S2", "London customer summit moved from June to September", s2),
        Story("S3", "USD weakened in Q3; US revenue under budget in GBP, on plan in USD", {}),
        Story("S4", "Customer Operations hiring freeze; support costs under budget", s4),
    ]


def budget_amount(key, month: int) -> float:
    ent, cc, acct = key
    base = BASE[key]
    if acct.startswith("4"):
        base *= (1 + REVENUE_GROWTH) ** (month - 1)
    return round(base, 2)


def build_ledger():
    rng = random.Random(SEED)
    stories = _story_multipliers()
    story_mult = {}
    for s in stories:
        story_mult.update(s.multipliers)

    budget_rows, actual_rows = [], []
    for key in sorted(BASE):
        ent, cc, acct = key
        for m in BUDGET_MONTHS:
            b = budget_amount(key, m)
            budget_rows.append({"entity_code": ent, "cost_centre_code": cc, "account_code": acct,
                                "period": f"{YEAR}-{m:02d}-01", "amount_local": b})
            if m in ACTUAL_MONTHS:
                mult = story_mult.get((ent, cc, acct, m))
                if mult is None:
                    mult = 1 + rng.uniform(-NOISE, NOISE)
                actual_rows.append({"entity_code": ent, "cost_centre_code": cc, "account_code": acct,
                                    "period": f"{YEAR}-{m:02d}-01",
                                    "amount_local": round(b * mult, 2)})
    return budget_rows, actual_rows, stories


def fx_rows():
    rows = []
    for m in BUDGET_MONTHS:
        rows.append({"currency": "GBP", "period": f"{YEAR}-{m:02d}-01", "rate_type": "budget", "to_gbp": 1.0})
        rows.append({"currency": "USD", "period": f"{YEAR}-{m:02d}-01", "rate_type": "budget", "to_gbp": BUDGET_USD_GBP})
        if m in ACTUAL_MONTHS:
            rows.append({"currency": "GBP", "period": f"{YEAR}-{m:02d}-01", "rate_type": "actual", "to_gbp": 1.0})
            rows.append({"currency": "USD", "period": f"{YEAR}-{m:02d}-01", "rate_type": "actual", "to_gbp": ACTUAL_USD_GBP[m]})
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --- Aggregation helpers (GBP), mirrors the dbt fact so commentary matches the model ---

def to_gbp(row, scenario):
    ent_ccy = {e[0]: e[2] for e in ENTITIES}[row["entity_code"]]
    if ent_ccy == "GBP":
        return row["amount_local"]
    m = int(row["period"][5:7])
    rate = BUDGET_USD_GBP if scenario == "budget" else ACTUAL_USD_GBP[m]
    return row["amount_local"] * rate


def agg(rows, scenario, months, **filters):
    acct_type = {a[0]: a[2] for a in ACCOUNTS}
    total = 0.0
    for r in rows:
        if int(r["period"][5:7]) not in months:
            continue
        if any(
            (k == "account_type" and acct_type[r["account_code"]] != v)
            or (k != "account_type" and r[k] != v)
            for k, v in filters.items()
        ):
            continue
        total += to_gbp(r, scenario)
    return round(total, 2)


def gbp(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}£{abs(x) / 1000:,.0f}k"


# --- Corpus ---

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def commentary(budget, actual) -> dict[int, str]:
    docs = {}
    for m in ACTUAL_MONTHS:
        mm = [m]
        rev_a, rev_b = agg(actual, "actual", mm, account_type="revenue"), agg(budget, "budget", mm, account_type="revenue")
        opex_a, opex_b = agg(actual, "actual", mm, account_type="opex"), agg(budget, "budget", mm, account_type="opex")
        name = MONTH_NAMES[m - 1]
        paras = [
            f"# Management commentary: {name} {YEAR}",
            "",
            f"Prepared by Group FP&A, Larkspur Ltd. Reporting currency GBP. "
            f"Figures are month actuals against the FY{YEAR} budget.",
            "",
            "## Headline",
            "",
            f"Revenue was {gbp(rev_a)} against a budget of {gbp(rev_b)} "
            f"({gbp(rev_a - rev_b)} variance). Operating expenses were {gbp(opex_a)} "
            f"against {gbp(opex_b)} ({gbp(opex_a - opex_b)} variance).",
            "",
            "## Variance notes",
            "",
        ]
        notes = []
        if m in (4, 5, 6):
            mk_a = agg(actual, "actual", mm, cost_centre_code="CC200", account_code="6400")
            mk_b = agg(budget, "budget", mm, cost_centre_code="CC200", account_code="6400")
            line = (f"Marketing programmes were {gbp(mk_a)} against {gbp(mk_b)}. ")
            if m == 6:
                line += ("The London customer summit has moved from June to September after the venue "
                         "contract was renegotiated. Summit costs will now land in September; this is a "
                         "timing difference, not a saving, and the full-year forecast is unchanged.")
            else:
                line += ("Spend is phased lower ahead of the summit, which is now being re-planned. "
                         "Treat the underspend as timing.")
            notes.append(line)
        if m in (7, 8, 9):
            c_a = agg(actual, "actual", mm, cost_centre_code="CC300", account_code="6100")
            c_b = agg(budget, "budget", mm, cost_centre_code="CC300", account_code="6100")
            notes.append(
                f"Engineering contractor spend was {gbp(c_a)} against a budget of {gbp(c_b)}. "
                "The data-centre exit and platform migration was brought forward from FY2027 to Q3 "
                "after the colocation provider gave notice. The CTO approved the contractor ramp in June; "
                "it is funded from the FY2027 migration budget and is expected to end in November. "
                "Cloud hosting is also running above budget while old and new platforms run in parallel."
            )
            us_rev_a = agg(actual, "actual", mm, entity_code="US01", account_type="revenue")
            us_rev_b = agg(budget, "budget", mm, entity_code="US01", account_type="revenue")
            notes.append(
                f"US revenue was {gbp(us_rev_a)} in GBP against a budget of {gbp(us_rev_b)}. "
                f"The shortfall is currency, not trading: the dollar weakened to around "
                f"{ACTUAL_USD_GBP[m]:.3f} GBP per USD against the budget rate of {BUDGET_USD_GBP:.2f}. "
                "In constant currency, US revenue is on plan."
            )
            co_a = agg(actual, "actual", mm, cost_centre_code="CC400")
            co_b = agg(budget, "budget", mm, cost_centre_code="CC400")
            notes.append(
                f"Customer Operations costs were {gbp(co_a)} against {gbp(co_b)}. The hiring freeze "
                "announced in July has left four support roles and two back-office roles vacant, so both "
                "support staff costs and salaries and benefits are below budget. Service levels are being "
                "watched weekly; backfill is planned for Q4 once the freeze lifts."
            )
        if m == 9:
            mk_a = agg(actual, "actual", mm, cost_centre_code="CC200", account_code="6400")
            mk_b = agg(budget, "budget", mm, cost_centre_code="CC200", account_code="6400")
            notes.append(
                f"Marketing programmes were {gbp(mk_a)} against {gbp(mk_b)}, as the London customer "
                "summit moved from June took place this month. Year to date, marketing programmes "
                "are close to budget."
            )
        if not notes:
            notes.append("No variances above the reporting threshold this month. "
                         "All cost centres are within 5% of budget.")
        paras += [n + "\n" for n in notes]
        docs[m] = "\n".join(paras).rstrip() + "\n"
    return docs


def stories_truth(budget, actual, stories) -> dict:
    q2, q3 = [4, 5, 6], [7, 8, 9]
    return {
        "company": "Larkspur Ltd (fictional)",
        "reporting_currency": "GBP",
        "stories": [
            {
                "id": "S1", "title": stories[0].title,
                "where": {"cost_centre": "Engineering", "account": "Contractors", "entity": "UK01", "period": "Q3"},
                "actual_gbp": agg(actual, "actual", q3, cost_centre_code="CC300", account_code="6100"),
                "budget_gbp": agg(budget, "budget", q3, cost_centre_code="CC300", account_code="6100"),
                "source_docs": ["commentary/2026-07.md", "commentary/2026-08.md", "commentary/2026-09.md",
                                "notes/engineering_q3_migration.md"],
            },
            {
                "id": "S2", "title": stories[1].title,
                "where": {"cost_centre": "Marketing", "account": "Marketing programmes", "period": "Q2 and September"},
                "q2_actual_gbp": agg(actual, "actual", q2, cost_centre_code="CC200", account_code="6400"),
                "q2_budget_gbp": agg(budget, "budget", q2, cost_centre_code="CC200", account_code="6400"),
                "source_docs": ["commentary/2026-06.md", "commentary/2026-09.md"],
            },
            {
                "id": "S3", "title": stories[2].title,
                "where": {"entity": "US01", "account_type": "revenue", "period": "Q3"},
                "actual_gbp": agg(actual, "actual", q3, entity_code="US01", account_type="revenue"),
                "budget_gbp": agg(budget, "budget", q3, entity_code="US01", account_type="revenue"),
                "source_docs": ["commentary/2026-07.md", "budget/fy2026_budget_memo.md"],
            },
            {
                "id": "S4", "title": stories[3].title,
                "where": {"cost_centre": "Customer Operations", "period": "Q3"},
                "actual_gbp": agg(actual, "actual", q3, cost_centre_code="CC400"),
                "budget_gbp": agg(budget, "budget", q3, cost_centre_code="CC400"),
                "source_docs": ["commentary/2026-07.md"],
            },
        ],
        "q3_opex": {
            "actual_gbp": agg(actual, "actual", q3, account_type="opex"),
            "budget_gbp": agg(budget, "budget", q3, account_type="opex"),
        },
    }


def main():
    budget, actual, stories = build_ledger()

    write_csv(SEEDS / "raw_entities.csv",
              [{"entity_code": c, "entity_name": n, "currency": ccy} for c, n, ccy in ENTITIES])
    write_csv(SEEDS / "raw_cost_centres.csv",
              [{"cost_centre_code": c, "cost_centre_name": n, "owner": o} for c, n, o in COST_CENTRES])
    write_csv(SEEDS / "raw_accounts.csv",
              [{"account_code": c, "account_name": n, "account_type": t} for c, n, t in ACCOUNTS])
    write_csv(SEEDS / "raw_fx_rates.csv", fx_rows())
    write_csv(SEEDS / "raw_budget.csv", budget)
    write_csv(SEEDS / "raw_gl_actuals.csv", actual)

    out = CORPUS / "commentary"
    out.mkdir(parents=True, exist_ok=True)
    for m, text in commentary(budget, actual).items():
        (out / f"{YEAR}-{m:02d}.md").write_text(text)

    truth = stories_truth(budget, actual, stories)
    (CORPUS / "stories.yaml").write_text(
        "# Ground truth for the planted stories. Generated; do not edit by hand.\n"
        + yaml.safe_dump(truth, sort_keys=False)
    )
    print(f"budget rows: {len(budget)}, actual rows: {len(actual)}, "
          f"commentary docs: {len(list(out.glob('*.md')))}")


if __name__ == "__main__":
    main()
