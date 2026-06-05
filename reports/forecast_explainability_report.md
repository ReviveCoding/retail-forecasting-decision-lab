# Forecast Explanation Grounding Report

## What was tested

Generated forecast explanations and evaluated numeric consistency, driver support, unsupported claims, and negative-control detection.

## Baseline

Feature importance alone does not verify whether generated explanations are faithful.

## Enhanced method

The grounding evaluator checks explanation text against forecast outputs and deliberately flawed negative controls.

## Key results

## Metrics

```json
{'n_explanations': 30, 'n_negative_controls': 5, 'numeric_consistency_rate': 0.8333333333333334, 'driver_direction_correctness_rate': 0.8333333333333334, 'uncertainty_disclosure_rate': 0.8333333333333334, 'unsupported_claim_rate': 0.16666666666666666, 'negative_control_detection_rate': 1.0, 'mean_groundedness_score': 0.8333333333333334}
```

## Examples

| case_type   | item_store_id    | date                |   actual |   tree_forecast |   corrected_forecast | top_global_feature   | explanation                                                                                                                                                                                                                                             |
|:------------|:-----------------|:--------------------|---------:|----------------:|---------------------:|:---------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| supported   | FOODS_1_027_CA_2 | 2016-05-21 00:00:00 |       23 |         4.49031 |             2.87587  | rolling_demand_state | Corrected forecast is lower than the tree-ensemble forecast for FOODS_1_027_CA_2 on 2016-05-21 because residual correction and demand-state features were applied. Top global driver: rolling_demand_state. tree=4.49, corrected=2.88, actual=23.00.    |
| supported   | FOODS_1_012_CA_2 | 2016-05-04 00:00:00 |       28 |        10.7724  |             9.33668  | rolling_demand_state | Corrected forecast is lower than the tree-ensemble forecast for FOODS_1_012_CA_2 on 2016-05-04 because residual correction and demand-state features were applied. Top global driver: rolling_demand_state. tree=10.77, corrected=9.34, actual=28.00.   |
| supported   | FOODS_1_015_CA_1 | 2016-05-21 00:00:00 |       16 |         1.17792 |             0.998588 | rolling_demand_state | Corrected forecast is lower than the tree-ensemble forecast for FOODS_1_015_CA_1 on 2016-05-21 because residual correction and demand-state features were applied. Top global driver: rolling_demand_state. tree=1.18, corrected=1.00, actual=16.00.    |
| supported   | FOODS_1_012_CA_2 | 2016-05-01 00:00:00 |       30 |        13.4414  |            15.0233   | rolling_demand_state | Corrected forecast is higher than the tree-ensemble forecast for FOODS_1_012_CA_2 on 2016-05-01 because residual correction and demand-state features were applied. Top global driver: rolling_demand_state. tree=13.44, corrected=15.02, actual=30.00. |
| supported   | FOODS_1_019_CA_2 | 2016-04-28 00:00:00 |        0 |        11.0607  |            13.774    | rolling_demand_state | Corrected forecast is higher than the tree-ensemble forecast for FOODS_1_019_CA_2 on 2016-04-28 because residual correction and demand-state features were applied. Top global driver: rolling_demand_state. tree=11.06, corrected=13.77, actual=0.00.  |

## Interpretation

Negative-control detection demonstrates that the evaluator can reject unsupported narratives instead of only scoring generated text as perfect.

## Claim boundary

No external LLM API is required in MVP; this is an LLM-ready deterministic grounding layer.
