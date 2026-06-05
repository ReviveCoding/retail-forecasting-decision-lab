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
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | service_level       |        0.972337 |          0.974432 |       0.00209444 |       0.00215402 |  0.00125375 |  0.0030613  |  0.000979384 |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | total_cost          |        0.933862 |          0.948851 |       0.0149889  |       0.0160505  |  0.00346437 |  0.0266281  |  0.0108061   |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | demand_coverage_f1  |        0.392248 |          0.389239 |      -0.00300906 |      -0.00767131 | -0.00417338 | -0.00141177 |  0.00151264  |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | inventory_precision |        0.344683 |          0.339652 |      -0.00503092 |      -0.0145958  | -0.00670413 | -0.00278441 |  0.00179104  |       300 |           60 | stable          |
| offline_ab_policy_readout | item-store-week | tree_ensemble_policy | risk_safe_corrected_policy | overstock_units     |        3.06512  |          3.17794  |       0.112822   |       0.0368085  |  0.0481847  |  0.175757   |  0.048027    |       300 |           60 | stable          |

## Interpretation

This is an experiment-style readout for portfolio evidence; CI direction and guardrails inform LAUNCH/ITERATE/HOLD/BLOCK.

## Claim boundary

This is not a randomized production A/B test.
