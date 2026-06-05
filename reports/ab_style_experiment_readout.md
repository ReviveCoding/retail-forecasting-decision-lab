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
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | service_level       |        0.972337 |          0.974445 |       0.00210795 |       0.00216792 |  0.0012463  |  0.00306068 |  0.000981443 |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | total_cost          |        0.933862 |          0.949527 |       0.0156649  |       0.0167743  |  0.00364719 |  0.0283229  |  0.0111682   |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | demand_coverage_f1  |        0.392248 |          0.389174 |      -0.00307352 |      -0.00783567 | -0.00428575 | -0.00144117 |  0.00152791  |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | inventory_precision |        0.344683 |          0.339573 |      -0.00511069 |      -0.0148272  | -0.00680174 | -0.00280446 |  0.00180033  |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | overstock_units     |        3.06512  |          3.18189  |       0.116771   |       0.0380966  |  0.0495635  |  0.183011   |  0.0496231   |       300 |           60 | stable          |

## Interpretation

This is an experiment-style readout for portfolio evidence; CI direction and guardrails inform LAUNCH/ITERATE/HOLD/BLOCK.

## Claim boundary

This is not a randomized production A/B test.
