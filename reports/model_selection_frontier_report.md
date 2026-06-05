# Model Selection Frontier Report

## Executive decision

Recommended decision: **ITERATE**

This report separates the point-forecast frontier from the inventory-policy frontier. The accuracy-optimal model is not automatically treated as the launch policy; policy selection is evaluated through stockout, overstock, total-cost, and coverage guardrails.

## Current recommendation

- Accuracy-optimal forecast: `learned_abc_gpu_residual`
- Release-gate policy: `learned_abc_release_safe_policy`
- WAPE improvement versus tree baseline: `7.75%`
- Remaining iteration item: `under_forecast_improved`

## Release-gate guardrails

| guardrail                           | status   |
|:------------------------------------|:---------|
| wape_improved                       | pass     |
| under_forecast_improved             | fail     |
| stockout_not_worse                  | pass     |
| overstock_guardrail_observed        | pass     |
| total_cost_guardrail_observed       | pass     |
| worst_slice_wape_guardrail_observed | pass     |
| ab_ci_stable_if_required            | pass     |

## Frontier summary

| frontier_role                 | candidate                       | layer    |       wape |      rmse |       mae |   mean_bias |   under_forecast_rate |   stockout_rate |   total_cost |   demand_coverage_f1 | decision_use                                                                   |
|:------------------------------|:--------------------------------|:---------|-----------:|----------:|----------:|------------:|----------------------:|----------------:|-------------:|---------------------:|:-------------------------------------------------------------------------------|
| Baseline forecast             | tree_ensemble                   | forecast |   0.78359  |   2.81444 |   1.74862 |   0.38278   |              0.3375   |      nan        |   nan        |           nan        | Reference tree-ensemble model.                                                 |
| Accuracy-optimal forecast     | learned_abc_gpu_residual        | forecast |   0.722877 |   2.75485 |   1.61314 |  -0.156777  |              0.426786 |      nan        |   nan        |           nan        | Lowest WAPE among evaluated point forecasts.                                   |
| RMSE-optimal forecast         | learned_abc_gpu_residual        | forecast |   0.722877 |   2.75485 |   1.61314 |  -0.156777  |              0.426786 |      nan        |   nan        |           nan        | Lowest RMSE among evaluated point forecasts.                                   |
| Bias-balanced forecast        | guarded_gpu_residual            | forecast |   0.734004 |   2.8008  |   1.63797 |  -0.0810581 |              0.414881 |      nan        |   nan        |           nan        | Lowest absolute mean bias, tie-broken by WAPE.                                 |
| Under-forecast-safe forecast  | release_safe_gpu_residual       | forecast |   0.794044 |   2.84307 |   1.77195 |   0.445909  |              0.327381 |      nan        |   nan        |           nan        | Lowest under-forecast rate, tie-broken by WAPE.                                |
| Baseline policy               | tree_ensemble_policy            | policy   | nan        | nan       | nan       | nan         |            nan        |        0.111905 |     0.95867  |             0.389963 | Reference inventory policy at the gate slice.                                  |
| Release-gate treatment policy | learned_abc_release_safe_policy | policy   | nan        | nan       | nan       | nan         |            nan        |        0.109524 |     0.964974 |             0.388676 | Policy currently used by gpu_guarded_release_gate.json.                        |
| Stockout-safe policy          | risk_safe_corrected_policy      | policy   | nan        | nan       | nan       | nan         |            nan        |        0.105952 |     0.974617 |             0.386939 | Lowest stockout rate at the gate slice.                                        |
| Coverage-F1 policy            | learned_abc_gpu_residual_policy | policy   | nan        | nan       | nan       | nan         |            nan        |        0.148214 |     0.909464 |             0.404416 | Highest inventory demand-coverage F1 at the gate slice.                        |
| Cost-balanced policy          | tree_ensemble_policy            | policy   | nan        | nan       | nan       | nan         |            nan        |        0.111905 |     0.95867  |             0.389963 | Lowest total cost among feasible policies under baseline-relative constraints. |

## Forecast candidates ranked by WAPE

| model_name                |   n_obs |    rmse |     mae |     wape |   mean_bias |   under_forecast_rate |   over_forecast_rate |
|:--------------------------|--------:|--------:|--------:|---------:|------------:|----------------------:|---------------------:|
| learned_abc_gpu_residual  |    1680 | 2.75485 | 1.61314 | 0.722877 |  -0.156777  |              0.426786 |             0.560119 |
| guarded_gpu_residual      |    1680 | 2.8008  | 1.63797 | 0.734004 |  -0.0810581 |              0.414881 |             0.578571 |
| moving_average            |    1680 | 2.86722 | 1.66671 | 0.746885 |  -0.109141  |              0.41369  |             0.576786 |
| corrected                 |    1680 | 2.75624 | 1.69372 | 0.758991 |   0.186047  |              0.357738 |             0.62619  |
| tree_ensemble             |    1680 | 2.81444 | 1.74862 | 0.78359  |   0.38278   |              0.3375   |             0.6625   |
| release_safe_gpu_residual |    1680 | 2.84307 | 1.77195 | 0.794044 |   0.445909  |              0.327381 |             0.672619 |
| seasonal_naive            |    1680 | 3.51316 | 2.00893 | 0.90024  |  -0.250595  |              0.382143 |             0.297619 |

## Policy candidates at lead_time=7, safety_stock_factor=0.5

| forecast_model                   |   lead_time |   safety_stock_factor |   n_obs |   stockout_rate |   service_level |   overstock_units |   holding_cost |   lost_sales_proxy |   total_cost |   labor_planning_error_proxy |   demand_recall |   inventory_precision |   demand_coverage_f1 |
|:---------------------------------|------------:|----------------------:|--------:|----------------:|----------------:|------------------:|---------------:|-------------------:|-------------:|-----------------------------:|----------------:|----------------------:|---------------------:|
| risk_safe_corrected_policy       |           7 |                   0.5 |    1680 |        0.105952 |        0.974531 |           3.27591 |       0.818979 |           0.155638 |     0.974617 |                    0.0359585 |        0.974531 |              0.337412 |             0.386939 |
| learned_abc_release_safe_policy  |           7 |                   0.5 |    1680 |        0.109524 |        0.972915 |           3.19197 |       0.797992 |           0.166981 |     0.964974 |                    0.0351692 |        0.972915 |              0.340501 |             0.388676 |
| release_safe_gpu_residual_policy |           7 |                   0.5 |    1680 |        0.110119 |        0.973301 |           3.20811 |       0.802027 |           0.16442  |     0.966447 |                    0.035439  |        0.973301 |              0.339535 |             0.388141 |
| tree_ensemble_policy             |           7 |                   0.5 |    1680 |        0.111905 |        0.972272 |           3.15138 |       0.787846 |           0.170824 |     0.95867  |                    0.0349724 |        0.972272 |              0.342604 |             0.389963 |
| corrected_policy                 |           7 |                   0.5 |    1680 |        0.122024 |        0.968766 |           2.9744  |       0.743599 |           0.19057  |     0.93417  |                    0.0338745 |        0.968766 |              0.356573 |             0.399709 |
| guarded_gpu_residual_policy      |           7 |                   0.5 |    1680 |        0.138095 |        0.961523 |           2.74332 |       0.685831 |           0.226601 |     0.912431 |                    0.0327593 |        0.961523 |              0.368526 |             0.404346 |
| learned_abc_gpu_residual_policy  |           7 |                   0.5 |    1680 |        0.148214 |        0.957917 |           2.68037 |       0.670093 |           0.239371 |     0.909464 |                    0.0322627 |        0.957917 |              0.372265 |             0.404416 |
| seasonal_naive_policy            |           7 |                   0.5 |    1680 |        0.18869  |        0.926871 |           2.70794 |       0.676984 |           0.360753 |     1.03774  |                    0.0401786 |        0.926871 |              0.394923 |             0.397325 |

## Interpretation

- `learned_abc_gpu_residual` is the accuracy-oriented forecast candidate because it minimizes WAPE in the current run.
- `learned_abc_release_safe_policy` is the release-gate treatment policy because it separates inventory ordering from the raw accuracy-optimal forecast.
- The decision remains ITERATE rather than LAUNCH because the under-forecast guardrail is still failing, even though the stockout, overstock, total-cost, and worst-slice guardrails pass.

## Resume-safe claim boundary

Safe claim: built a leakage-safe forecasting decision framework with an accuracy frontier, policy frontier, release gates, and ABC-style demand coverage correction.

Do not claim production launch, real-world stockout reduction, or Amazon production impact.