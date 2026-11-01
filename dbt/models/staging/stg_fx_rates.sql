select currency, period, rate_type, to_gbp
from {{ ref('raw_fx_rates') }}
