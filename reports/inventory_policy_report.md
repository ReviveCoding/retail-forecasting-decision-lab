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
| corrected_policy |           1 |                   0   |    1680 |       0.357738  |        0.845073 |          0.939886 |       0.234971 |          0.753839  |     0.98881  |                    0.0338745 |        0.845073 |              0.527497 |             0.44272  |
| corrected_policy |           1 |                   0.5 |    1680 |       0.232143  |        0.916721 |          1.63151  |       0.407878 |          0.463596  |     0.871473 |                    0.0338745 |        0.916721 |              0.447138 |             0.435911 |
| corrected_policy |           1 |                   1   |    1680 |       0.16131   |        0.953486 |          2.42611  |       0.606528 |          0.27633   |     0.882858 |                    0.0338745 |        0.953486 |              0.388311 |             0.4159   |
| corrected_policy |           3 |                   0   |    1680 |       0.357738  |        0.845073 |          0.939886 |       0.234971 |          0.753839  |     0.98881  |                    0.0338745 |        0.845073 |              0.527497 |             0.44272  |
| corrected_policy |           3 |                   0.5 |    1680 |       0.17619   |        0.945449 |          2.2059   |       0.551476 |          0.319212  |     0.870688 |                    0.0338745 |        0.945449 |              0.402743 |             0.42206  |
| corrected_policy |           3 |                   1   |    1680 |       0.0797619 |        0.980206 |          3.70869  |       0.927171 |          0.121347  |     1.04852  |                    0.0338745 |        0.980206 |              0.320141 |             0.376418 |
| corrected_policy |           7 |                   0   |    1680 |       0.357738  |        0.845073 |          0.939886 |       0.234971 |          0.753839  |     0.98881  |                    0.0338745 |        0.845073 |              0.527497 |             0.44272  |
| corrected_policy |           7 |                   0.5 |    1680 |       0.122024  |        0.968766 |          2.9744   |       0.743599 |          0.19057   |     0.93417  |                    0.0338745 |        0.968766 |              0.356573 |             0.399709 |
| corrected_policy |           7 |                   1   |    1680 |       0.0309524 |        0.991568 |          5.43807  |       1.35952  |          0.0564663 |     1.41598  |                    0.0338745 |        0.991568 |              0.256723 |             0.326997 |
| corrected_policy |          14 |                   0   |    1680 |       0.357738  |        0.845073 |          0.939886 |       0.234971 |          0.753839  |     0.98881  |                    0.0338745 |        0.845073 |              0.527497 |             0.44272  |

## Interpretation

Stockout, overstock, service level, total cost, Demand Recall, Inventory Precision, and Coverage F1 expose business trade-offs hidden by WAPE.

## Claim boundary

Inventory, lead-time, labor, and cost metrics are synthetic/proxy overlays.
