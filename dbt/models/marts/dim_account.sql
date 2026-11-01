select account_code, account_name, account_type
from {{ ref('raw_accounts') }}
