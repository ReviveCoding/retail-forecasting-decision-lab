# GPU Residual Corrector Smoke Report

## Purpose

This experiment evaluates whether a CUDA-backed PyTorch MLP can learn residual structure on the enriched forecast artifact and improve a tree-ensemble forecast.

## Metrics

| model                        | device   |   n_test |     wape |    rmse |    mean_bias |   under_forecast_rate |   train_seconds |   inference_seconds |
|:-----------------------------|:---------|---------:|---------:|--------:|-------------:|----------------------:|----------------:|--------------------:|
| tree_ensemble                | cpu      |      252 | 0.68948  | 3.02234 |  0.209212    |              0.345238 |          0      |                   0 |
| current_residual_corrected   | cpu      |      252 | 0.694348 | 3.10019 | -0.000878894 |              0.357143 |          0      |                   0 |
| torch_gpu_residual_corrected | cuda     |      252 | 0.651888 | 3.07841 | -0.792444    |              0.496032 |          0.3666 |                   0 |

## Interpretation

This is an exploratory smoke experiment. It validates the GPU residual-correction training path using feature-rich generated artifacts. Final resume-safe claims should come from a leakage-safe integration inside the main train/validation/test pipeline.

## Claim boundary

Do not describe this as a production or final launch result. Use it as GPU ML feasibility evidence until integrated into the core pipeline.