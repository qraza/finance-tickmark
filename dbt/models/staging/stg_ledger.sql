-- Actuals and budget in one long table, local currency.
select entity_code, cost_centre_code, account_code, period, 'actual' as scenario, amount_local
from {{ ref('raw_gl_actuals') }}
union all
select entity_code, cost_centre_code, account_code, period, 'budget' as scenario, amount_local
from {{ ref('raw_budget') }}
