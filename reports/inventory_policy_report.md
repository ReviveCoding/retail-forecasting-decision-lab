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
| corrected_policy |           1 |                   0   |    1680 |       0.357143  |        0.845369 |          0.940988 |       0.235247 |          0.752908  |     0.988155 |                    0.0338779 |        0.845369 |              0.527373 |             0.442934 |
| corrected_policy |           1 |                   0.5 |    1680 |       0.231548  |        0.91688  |          1.63343  |       0.408357 |          0.46348   |     0.871837 |                    0.0338779 |        0.91688  |              0.446912 |             0.435891 |
| corrected_policy |           1 |                   1   |    1680 |       0.16131   |        0.953514 |          2.42845  |       0.607113 |          0.276635  |     0.883748 |                    0.0338779 |        0.953514 |              0.388086 |             0.415773 |
| corrected_policy |           3 |                   0   |    1680 |       0.357143  |        0.845369 |          0.940988 |       0.235247 |          0.752908  |     0.988155 |                    0.0338779 |        0.845369 |              0.527373 |             0.442934 |
| corrected_policy |           3 |                   0.5 |    1680 |       0.174405  |        0.945505 |          2.2082   |       0.55205  |          0.319474  |     0.871524 |                    0.0338779 |        0.945505 |              0.402517 |             0.421953 |
| corrected_policy |           3 |                   1   |    1680 |       0.0797619 |        0.980199 |          3.71106  |       0.927765 |          0.121691  |     1.04946  |                    0.0338779 |        0.980199 |              0.319994 |             0.376298 |
| corrected_policy |           7 |                   0   |    1680 |       0.357143  |        0.845369 |          0.940988 |       0.235247 |          0.752908  |     0.988155 |                    0.0338779 |        0.845369 |              0.527373 |             0.442934 |
| corrected_policy |           7 |                   0.5 |    1680 |       0.122024  |        0.96875  |          2.97689  |       0.744222 |          0.191028  |     0.935249 |                    0.0338779 |        0.96875  |              0.356361 |             0.399556 |
| corrected_policy |           7 |                   1   |    1680 |       0.0309524 |        0.991564 |          5.44022  |       1.36006  |          0.0565838 |     1.41664  |                    0.0338779 |        0.991564 |              0.256637 |             0.326913 |
| corrected_policy |          14 |                   0   |    1680 |       0.357143  |        0.845369 |          0.940988 |       0.235247 |          0.752908  |     0.988155 |                    0.0338779 |        0.845369 |              0.527373 |             0.442934 |

## Interpretation

Stockout, overstock, service level, total cost, Demand Recall, Inventory Precision, and Coverage F1 expose business trade-offs hidden by WAPE.

## Claim boundary

Inventory, lead-time, labor, and cost metrics are synthetic/proxy overlays.
