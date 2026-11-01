# Metric definitions

Generated from the dbt semantic layer. Do not edit by hand; edit dbt/models/marts/metrics.yml.

## Revenue (`revenue`)

Subscription and professional services revenue (accounts 4xxx). Actuals in GBP at the monthly actual FX rate.

Definition: Sum of actual revenue in GBP.

Owner: Priya Nair (Commercial)

## Revenue budget (`revenue_budget`)

Revenue budget in GBP at the budget FX rate, including months not yet closed.

Definition: Full-year phased budget for revenue.

Owner: Priya Nair (Commercial)

## Revenue budget (closed months) (`revenue_budget_to_date`)

Revenue budget for closed months only. The comparator for revenue_variance.

Definition: Budget restricted to months with actuals, so variances compare like for like.

Owner: Priya Nair (Commercial)

## Revenue variance (`revenue_variance`)

Actual minus budget for closed months. Positive is favourable.

Definition: Actual minus budget, closed months only. FP&A policy sign convention.

Owner: Priya Nair (Commercial)

## Revenue variance % (`revenue_variance_pct`)

Revenue variance as a fraction of budget for closed months.

Definition: Variance divided by budget, closed months only.

Owner: Priya Nair (Commercial)

## Cost of sales (`cogs`)

Cost of sales: cloud hosting and support staff (accounts 5xxx). Actuals in GBP at the monthly actual FX rate.

Definition: Sum of actual cost of sales in GBP.

Owner: Group FP&A

## Cost of sales budget (`cogs_budget`)

Cost of sales budget in GBP at the budget FX rate, including months not yet closed.

Definition: Full-year phased budget for cost of sales.

Owner: Group FP&A

## Cost of sales budget (closed months) (`cogs_budget_to_date`)

Cost of sales budget for closed months only. The comparator for cogs_variance.

Definition: Budget restricted to months with actuals, so variances compare like for like.

Owner: Group FP&A

## Cost of sales variance (`cogs_variance`)

Actual minus budget for closed months. Positive is adverse (overspend).

Definition: Actual minus budget, closed months only. FP&A policy sign convention.

Owner: Group FP&A

## Cost of sales variance % (`cogs_variance_pct`)

Cost of sales variance as a fraction of budget for closed months.

Definition: Variance divided by budget, closed months only.

Owner: Group FP&A

## Operating expenses (`opex`)

Operating expenses: salaries, contractors, software, travel, marketing, rent (accounts 6xxx). Actuals in GBP at the monthly actual FX rate.

Definition: Sum of actual operating expenses in GBP.

Owner: Group FP&A

## Operating expenses budget (`opex_budget`)

Operating expenses budget in GBP at the budget FX rate, including months not yet closed.

Definition: Full-year phased budget for operating expenses.

Owner: Group FP&A

## Operating expenses budget (closed months) (`opex_budget_to_date`)

Operating expenses budget for closed months only. The comparator for opex_variance.

Definition: Budget restricted to months with actuals, so variances compare like for like.

Owner: Group FP&A

## Operating expenses variance (`opex_variance`)

Actual minus budget for closed months. Positive is adverse (overspend).

Definition: Actual minus budget, closed months only. FP&A policy sign convention.

Owner: Group FP&A

## Operating expenses variance % (`opex_variance_pct`)

Operating expenses variance as a fraction of budget for closed months.

Definition: Variance divided by budget, closed months only.

Owner: Group FP&A

## Revenue (constant currency) (`revenue_constant_currency`)

Revenue actuals restated at the budget FX rate. Strips out FX translation so trading can be compared to budget.

Definition: Actual revenue in local currency translated at the budget rate (0.79 GBP per USD).

Owner: Group FP&A

## FX impact on revenue (`fx_impact_on_revenue`)

Reported revenue minus constant-currency revenue. Negative means currency reduced reported revenue.

Definition: Translation difference between actual and budget FX rates on revenue.

Owner: Group FP&A

## Gross profit (`gross_profit`)

Revenue minus cost of sales.

Definition: Revenue minus cost of sales.

Owner: Group FP&A

## EBITDA (`ebitda`)

Revenue minus cost of sales minus operating expenses. Larkspur has no D&A in this model, so EBITDA equals operating profit.

Definition: Revenue minus cost of sales minus operating expenses. Larkspur has no D&A in this model, so EBITDA equals operating profit.

Owner: Group FP&A

## EBITDA budget (closed months) (`ebitda_budget_to_date`)

Budgeted EBITDA for closed months.

Definition: Budgeted EBITDA for closed months.

Owner: Group FP&A

## EBITDA variance (`ebitda_variance`)

Actual minus budget EBITDA, closed months. Positive is favourable.

Definition: Actual minus budget EBITDA, closed months. Positive is favourable.

Owner: Group FP&A

## EBITDA margin (`ebitda_margin`)

EBITDA as a fraction of revenue.

Definition: EBITDA as a fraction of revenue.

Owner: Group FP&A
