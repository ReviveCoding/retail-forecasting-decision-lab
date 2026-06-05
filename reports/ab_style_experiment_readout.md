# Offline A/B-Style Experiment Readout

## What was tested

Compared control and treatment ordering policies using item-store-week units and clustered bootstrap confidence intervals.

## Baseline

Control policy uses the tree-ensemble forecast.

## Enhanced method

Treatment policy uses the risk-safe corrected ordering signal.

## Key results

| experiment_id             | unit_type       | control_policy       | treatment_policy           | metric_name         |   control_value |   treatment_value |   absolute_delta |   relative_delta |      ci_low |     ci_high |   mde_approx |   n_units |   n_clusters | decision_hint   |
|:--------------------------|:----------------|:---------------------|:---------------------------|:--------------------|----------------:|------------------:|-----------------:|-----------------:|------------:|------------:|-------------:|----------:|-------------:|:----------------|
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | stockout_flag       |        0.11     |          0.104921 |      -0.00507937 |      -0.046176   | -0.0095377  | -0.00111111 |  0.00444266  |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | service_level       |        0.972337 |          0.974436 |       0.00209804 |       0.00215773 |  0.00124119 |  0.00306376 |  0.000974946 |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | total_cost          |        0.933862 |          0.948968 |       0.0151064  |       0.0161762  |  0.00352825 |  0.0274992  |  0.0106552   |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | demand_coverage_f1  |        0.392248 |          0.389255 |      -0.00299291 |      -0.00763014 | -0.00412426 | -0.00137286 |  0.00150591  |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | inventory_precision |        0.344683 |          0.339663 |      -0.00502005 |      -0.0145642  | -0.00669222 | -0.00277609 |  0.0017773   |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | overstock_units     |        3.06512  |          3.17926  |       0.114147   |       0.0372408  |  0.0496561  |  0.17582    |  0.048145    |       300 |           60 | stable          |

## Interpretation

This is an experiment-style readout for portfolio evidence; CI direction and guardrails inform LAUNCH/ITERATE/HOLD/BLOCK.

## Claim boundary

This is not a randomized production A/B test.
