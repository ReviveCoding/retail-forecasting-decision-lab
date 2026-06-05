# Inventory Policy Report

## What was tested

Translated forecasts into synthetic ordering policies under lead-time and safety-stock assumptions.

## Baseline

The baseline policy orders from the tree-ensemble forecast plus safety stock.

## Enhanced method

Corrected and risk-safe corrected policies are evaluated as ordering policies, not as pure point forecasts.

## Key results

| forecast_model   |   lead_time |   safety_stock_factor |   n_obs |   stockout_rate |   service_level |   overstock_units |   holding_cost |   lost_sales_proxy |   total_cost |   labor_planning_error_proxy |   demand_recall |   inventory_precision |   demand_coverage_f1 |
|:-----------------|------------:|----------------------:|--------:|----------------:|----------------:|------------------:|---------------:|-------------------:|-------------:|-----------------------------:|----------------:|----------------------:|---------------------:|
| corrected_policy |           1 |                   0   |    1680 |       0.356548  |        0.845522 |          0.943299 |       0.235825 |          0.75239   |     0.988215 |                    0.0339138 |        0.845522 |              0.527006 |             0.442703 |
| corrected_policy |           1 |                   0.5 |    1680 |       0.232143  |        0.917008 |          1.63555  |       0.408886 |          0.462768  |     0.871654 |                    0.0339138 |        0.917008 |              0.446771 |             0.435796 |
| corrected_policy |           1 |                   1   |    1680 |       0.161905  |        0.953542 |          2.43114  |       0.607785 |          0.276495  |     0.884281 |                    0.0339138 |        0.953542 |              0.387916 |             0.415631 |
| corrected_policy |           3 |                   0   |    1680 |       0.356548  |        0.845522 |          0.943299 |       0.235825 |          0.75239   |     0.988215 |                    0.0339138 |        0.845522 |              0.527006 |             0.442703 |
| corrected_policy |           3 |                   0.5 |    1680 |       0.174405  |        0.945565 |          2.21064  |       0.552659 |          0.319083  |     0.871742 |                    0.0339138 |        0.945565 |              0.402355 |             0.421827 |
| corrected_policy |           3 |                   1   |    1680 |       0.0797619 |        0.980229 |          3.71364  |       0.928411 |          0.121441  |     1.04985  |                    0.0339138 |        0.980229 |              0.319906 |             0.376217 |
| corrected_policy |           7 |                   0   |    1680 |       0.356548  |        0.845522 |          0.943299 |       0.235825 |          0.75239   |     0.988215 |                    0.0339138 |        0.845522 |              0.527006 |             0.442703 |
| corrected_policy |           7 |                   0.5 |    1680 |       0.122619  |        0.968777 |          2.97946  |       0.744866 |          0.190773  |     0.935639 |                    0.0339138 |        0.968777 |              0.356236 |             0.399445 |
| corrected_policy |           7 |                   1   |    1680 |       0.0309524 |        0.991582 |          5.44285  |       1.36071  |          0.0563843 |     1.4171   |                    0.0339138 |        0.991582 |              0.256576 |             0.326853 |
| corrected_policy |          14 |                   0   |    1680 |       0.356548  |        0.845522 |          0.943299 |       0.235825 |          0.75239   |     0.988215 |                    0.0339138 |        0.845522 |              0.527006 |             0.442703 |

## Interpretation

Stockout, overstock, service level, total cost, Demand Recall, Inventory Precision, and Coverage F1 expose business trade-offs hidden by WAPE.

## Claim boundary

Inventory, lead-time, labor, and cost metrics are synthetic/proxy overlays.
