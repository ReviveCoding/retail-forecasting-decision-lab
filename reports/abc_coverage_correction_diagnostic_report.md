# ABC-Style Selective Demand Coverage Correction Diagnostic

## Purpose

This diagnostic tests whether ABC-style selective demand inflation can improve the trade-off between WAPE, demand coverage recall, and order precision.

The script does not modify the main pipeline. It splits the generated holdout artifact into a diagnostic guard half and diagnostic evaluation half.

## Selected guard parameters

| candidate      |   n_rows |   actual_sum |   pred_sum |     wape |   demand_coverage_recall |   order_precision |   demand_coverage_f1 |   under_coverage_amount |   over_coverage_amount |   under_coverage_rate |   over_coverage_rate |   mean_bias |   alpha |   risk_power |   risk_quantile_gate |   risk_threshold |   score |
|:---------------|---------:|-------------:|-----------:|---------:|-------------------------:|------------------:|---------------------:|------------------------:|-----------------------:|----------------------:|---------------------:|------------:|--------:|-------------:|---------------------:|-----------------:|--------:|
| abc_guard_grid |      900 |         2043 |    2058.37 | 0.712202 |                  0.64766 |          0.642825 |             0.645233 |                 719.831 |                735.197 |              0.388889 |             0.611111 |   0.0170733 |     0.4 |          0.5 |                 0.25 |         0.312653 |     155 |

## Diagnostic evaluation metrics

| candidate                   |   n_rows |   actual_sum |   pred_sum |     wape |   demand_coverage_recall |   order_precision |   demand_coverage_f1 |   under_coverage_amount |   over_coverage_amount |   under_coverage_rate |   over_coverage_rate |   mean_bias |
|:----------------------------|---------:|-------------:|-----------:|---------:|-------------------------:|------------------:|---------------------:|------------------------:|-----------------------:|----------------------:|---------------------:|------------:|
| pred_tree_eval              |      780 |         1706 |    2076.41 | 0.827808 |                 0.694656 |          0.570737 |             0.626629 |                 520.917 |                891.324 |              0.337179 |             0.662821 |   0.474881  |
| pred_guarded_gpu_eval       |      780 |         1706 |    1663.03 | 0.767128 |                 0.603842 |          0.619445 |             0.611544 |                 675.846 |                632.874 |              0.420513 |             0.569231 |  -0.0550927 |
| pred_release_safe_gpu_eval  |      780 |         1706 |    2113.68 | 0.838118 |                 0.700426 |          0.565329 |             0.625668 |                 511.073 |                918.756 |              0.329487 |             0.670513 |   0.522671  |
| pred_abc_selective_gpu_eval |      780 |         1706 |    1763.77 | 0.783232 |                 0.625314 |          0.604834 |             0.614904 |                 639.214 |                696.98  |              0.403846 |             0.596154 |   0.0740584 |

## Interpretation

- `pred_guarded_gpu_eval` is the WAPE-oriented GPU forecast.
- `pred_release_safe_gpu_eval` is the conservative floor-based forecast.
- `pred_abc_selective_gpu_eval` selectively moves the guarded forecast toward the conservative forecast only for high under-risk rows.

## Claim boundary

This is a diagnostic experiment. Do not use it as a resume performance claim until the same logic is integrated into the leakage-safe train-base / residual-fit / guard-validation / test pipeline.