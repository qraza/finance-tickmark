select cost_centre_code, cost_centre_name, owner
from {{ ref('raw_cost_centres') }}
