# ABC-Style Demand Coverage Reliability Report

## Purpose

This report adapts the ABC Bbox region-wise Recall, Precision, and F1 idea to retail demand forecasting.

Demand coverage recall measures how much actual demand is covered by a forecast or order signal.

Order precision measures how much forecasted or ordered demand is justified by actual demand.

Demand coverage F1 summarizes the coverage-risk trade-off between under-forecasting and over-forecasting.

## Source

`outputs\corrected_forecasts_sample.csv`

## Candidate metrics

| candidate                              |   n_rows |   actual_sum |   pred_sum |     wape |   demand_coverage_recall |   order_precision |   demand_coverage_f1 |   under_coverage_amount |   over_coverage_amount |   under_coverage_rate |   over_coverage_rate |   mean_bias |   coverage_recall_delta_vs_tree |   order_precision_delta_vs_tree |   coverage_f1_delta_vs_tree |   wape_delta_vs_tree |
|:---------------------------------------|---------:|-------------:|-----------:|---------:|-------------------------:|------------------:|---------------------:|------------------------:|-----------------------:|----------------------:|---------------------:|------------:|--------------------------------:|--------------------------------:|----------------------------:|---------------------:|
| pred_tree                              |     1680 |         3749 |    4392.07 | 0.78359  |                 0.693971 |          0.592362 |             0.639153 |                 1147.3  |                1790.38 |              0.3375   |             0.6625   |   0.38278   |                       0         |                      0          |                 0           |            0         |
| pred_release_safe_gpu_residual         |     1680 |         3749 |    4498.13 | 0.794044 |                 0.702888 |          0.585828 |             0.639041 |                 1113.87 |                1863    |              0.327381 |             0.672619 |   0.445909  |                       0.0089175 |                     -0.00653425 |                -0.000111876 |            0.0104542 |
| order_signal_release_safe_gpu_residual |     1680 |         3749 |    4498.13 | 0.794044 |                 0.702888 |          0.585828 |             0.639041 |                 1113.87 |                1863    |              0.327381 |             0.672619 |   0.445909  |                       0.0089175 |                     -0.00653425 |                -0.000111876 |            0.0104542 |
| pred_corrected                         |     1680 |         3749 |    4069.73 | 0.759871 |                 0.662839 |          0.610602 |             0.635649 |                 1264.01 |                1584.74 |              0.356548 |             0.627381 |   0.190909  |                      -0.0311311 |                      0.0182404  |                -0.00350367  |           -0.0237189 |
| pred_guarded_gpu_residual              |     1680 |         3749 |    3612.82 | 0.734004 |                 0.614836 |          0.638011 |             0.626209 |                 1443.98 |                1307.8  |              0.414881 |             0.578571 |  -0.0810581 |                      -0.0791345 |                      0.045649   |                -0.012944    |           -0.049586  |
| order_signal_guarded_gpu_residual      |     1680 |         3749 |    3612.82 | 0.734004 |                 0.614836 |          0.638011 |             0.626209 |                 1443.98 |                1307.8  |              0.414881 |             0.578571 |  -0.0810581 |                      -0.0791345 |                      0.045649   |                -0.012944    |           -0.049586  |
| pred_moving_average                    |     1680 |         3749 |    3565.64 | 0.746885 |                 0.602103 |          0.633066 |             0.617196 |                 1491.71 |                1308.36 |              0.41369  |             0.576786 |  -0.109141  |                      -0.0918671 |                      0.0407037  |                -0.0219567   |           -0.0367054 |
| pred_seasonal_naive                    |     1680 |         3749 |    3328    | 0.90024  |                 0.493732 |          0.55619  |             0.523103 |                 1898    |                1477    |              0.382143 |             0.297619 |  -0.250595  |                      -0.200239  |                     -0.036172   |                -0.11605     |            0.11665   |

## Interpretation guide

- Higher demand coverage recall means lower under-coverage risk.
- Higher order precision means lower over-coverage risk.
- Higher demand coverage F1 means a better balance between coverage and precision.
- A candidate can improve WAPE while still hurting coverage risk, so this report should be read together with forecast_metrics.csv and release_gate.json.

## Claim boundary

This is an offline public-data evaluation artifact, not evidence of production inventory impact.