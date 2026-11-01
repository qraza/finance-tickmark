-- One row per legal entity x cost centre x account x month, in GBP.
--   actual_gbp            actuals at the monthly actual rate (reported)
--   actual_gbp_cc         actuals at the budget rate (constant currency)
--   budget_gbp            full-year budget at the budget rate
--   budget_gbp_closed     budget only for months whose books are closed, so
--                         variances never compare a missing actual to a budget
with ledger as (
    select l.*, e.currency
    from {{ ref('stg_ledger') }} l
    join {{ ref('dim_legal_entity') }} e using (entity_code)
),

fx as (select * from {{ ref('stg_fx_rates') }}),

closed as (select distinct period from {{ ref('raw_gl_actuals') }}),

converted as (
    select
        ledger.entity_code,
        ledger.cost_centre_code,
        ledger.account_code,
        ledger.period,
        ledger.scenario,
        ledger.amount_local,
        ledger.amount_local * fx_budget.to_gbp as amount_gbp_budget_rate,
        ledger.amount_local * coalesce(fx_actual.to_gbp, fx_budget.to_gbp) as amount_gbp_actual_rate
    from ledger
    join fx as fx_budget
        on fx_budget.currency = ledger.currency
        and fx_budget.period = ledger.period
        and fx_budget.rate_type = 'budget'
    left join fx as fx_actual
        on fx_actual.currency = ledger.currency
        and fx_actual.period = ledger.period
        and fx_actual.rate_type = 'actual'
)

select
    md5(entity_code || '|' || cost_centre_code || '|' || account_code || '|' || cast(period as varchar)) as pnl_line_id,
    entity_code,
    cost_centre_code,
    account_code,
    period,
    period in (select period from closed) as is_closed_period,
    sum(case when scenario = 'actual' then amount_gbp_actual_rate end) as actual_gbp,
    sum(case when scenario = 'actual' then amount_gbp_budget_rate end) as actual_gbp_cc,
    sum(case when scenario = 'budget' then amount_gbp_budget_rate end) as budget_gbp,
    sum(case when scenario = 'budget' and period in (select period from closed)
             then amount_gbp_budget_rate end) as budget_gbp_closed
from converted
group by all
