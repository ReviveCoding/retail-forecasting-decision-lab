# Guarded GPU Residual Correction Report

## Purpose

This post-processing experiment evaluates whether GPU-predicted residuals can be clipped and scaled to preserve WAPE improvement while reducing business-risk side effects such as excessive under-forecasting or bias shift.

## Baselines and raw GPU correction

| model                            |     wape |    rmse |    mean_bias |   under_forecast_rate |   lambda |   clip_quantile |
|:---------------------------------|---------:|--------:|-------------:|----------------------:|---------:|----------------:|
| tree_ensemble                    | 0.68948  | 3.02234 |  0.209212    |              0.345238 |      nan |             nan |
| current_residual_corrected       | 0.694348 | 3.10019 | -0.000878901 |              0.357143 |      nan |             nan |
| torch_gpu_residual_corrected_raw | 0.651888 | 3.07841 | -0.792444    |              0.496032 |      nan |             nan |

## Selected guarded candidate

| model                                    |     wape |    rmse |   mean_bias |   under_forecast_rate |   lambda |   clip_quantile |   score |
|:-----------------------------------------|---------:|--------:|------------:|----------------------:|---------:|----------------:|--------:|
| guarded_torch_residual_lambda_0.2_q_0.95 | 0.670724 | 3.00561 |  0.00122396 |              0.400794 |      0.2 |            0.95 |    22.5 |

## Interpretation

Raw GPU residual correction improved WAPE but increased under-forecast risk. The guarded variant searches correction strength and residual clipping levels to find a more balanced correction policy.

## Claim boundary

This is still a post-hoc smoke experiment on generated artifacts. Resume-safe final claims should come from integrating guarded correction into the main train/validation/test pipeline.