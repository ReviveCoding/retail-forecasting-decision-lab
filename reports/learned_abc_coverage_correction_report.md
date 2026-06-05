# Learned ABC-Style Demand Coverage Correction Diagnostic

## Purpose

This diagnostic learns under-coverage and over-coverage targets from the generated forecasting artifact, then evaluates whether a learned ABC-style asymmetric correction improves the WAPE and coverage trade-off.

## Split

- fit rows: 720
- selection rows: 420
- evaluation rows: 540

## Selected parameters

| candidate               |   n_rows |   actual_sum |   pred_sum |     wape |   demand_coverage_recall |   order_precision |   demand_coverage_f1 |   under_coverage_amount |   over_coverage_amount |   under_coverage_rate |   over_coverage_rate |   mean_bias |   alpha |   beta |   cap_quantile |   under_cap |   over_cap |   score |
|:------------------------|---------:|-------------:|-----------:|---------:|-------------------------:|------------------:|---------------------:|------------------------:|-----------------------:|----------------------:|---------------------:|------------:|--------:|-------:|---------------:|------------:|-----------:|--------:|
| learned_abc_select_grid |      420 |          900 |    922.781 | 0.729629 |                 0.647842 |          0.631848 |             0.639745 |                 316.942 |                339.723 |              0.383333 |             0.616667 |   0.0542401 |     0.5 |    0.3 |           0.75 |     1.06135 |    1.06829 |     168 |

## Evaluation metrics

| candidate                  |   n_rows |   actual_sum |   pred_sum |     wape |   demand_coverage_recall |   order_precision |   demand_coverage_f1 |   under_coverage_amount |   over_coverage_amount |   under_coverage_rate |   over_coverage_rate |   mean_bias |
|:---------------------------|---------:|-------------:|-----------:|---------:|-------------------------:|------------------:|---------------------:|------------------------:|-----------------------:|----------------------:|---------------------:|------------:|
| pred_tree_eval             |      540 |         1252 |    1500.74 | 0.80319  |                 0.697741 |          0.582095 |             0.634693 |                 378.428 |                627.166 |              0.337037 |             0.662963 |   0.460626  |
| pred_guarded_gpu_eval      |      540 |         1252 |    1220.13 | 0.746032 |                 0.614257 |          0.630301 |             0.622175 |                 482.95  |                451.082 |              0.427778 |             0.557407 |  -0.0590162 |
| pred_release_safe_gpu_eval |      540 |         1252 |    1532.17 | 0.816032 |                 0.703874 |          0.575164 |             0.633043 |                 370.749 |                650.922 |              0.331481 |             0.668519 |   0.518839  |
| pred_learned_abc_gpu_eval  |      540 |         1252 |    1306.32 | 0.760328 |                 0.641528 |          0.614853 |             0.627907 |                 448.807 |                503.123 |              0.374074 |             0.625926 |   0.100584  |

## Claim boundary

This is a diagnostic experiment on generated holdout artifacts. It should be integrated into the main leakage-safe pipeline before being used as a performance claim.