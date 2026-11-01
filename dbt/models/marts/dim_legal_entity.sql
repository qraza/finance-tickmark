select entity_code, entity_name, currency
from {{ ref('raw_entities') }}
