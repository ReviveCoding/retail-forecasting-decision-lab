# Launch Decision Memo

## Decision: BLOCK

This decision compares a tree-ensemble point forecast with a residual-corrected forecast, then evaluates the risk-safe corrected ordering policy under a public-data, offline, synthetic-inventory benchmark.

## Key evidence
- WAPE relative improvement: 0.0303
- Under-forecast relative improvement: -0.0564
- Stockout delta, treatment minus control: -0.005952380952380959
- Overstock relative worsening: 0.039954291950351856
- Total-cost relative worsening: 0.016870710756267296

## Guardrails
- wape_improved: pass
- under_forecast_improved: fail
- stockout_not_worse: pass
- overstock_guardrail_observed: fail
- total_cost_guardrail_observed: pass
- worst_slice_wape_guardrail_observed: pass
- ab_ci_stable_if_required: pass

## Recommendation
Do not launch this treatment policy because a business guardrail failed materially.

## Claim boundary
This is not a production Amazon launch memo. It is a public-data, local-runnable measurement artifact.
