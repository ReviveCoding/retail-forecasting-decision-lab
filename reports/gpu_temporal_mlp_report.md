# GPU Temporal MLP Forecasting Smoke Report

## Purpose

This lightweight experiment validates that a PyTorch model can train and run inference on the retail forecasting feature mart. It is an optional GPU-backed extension and does not replace the main CPU-runnable pipeline.

## Metrics

| model              | device   |   n_test |     wape |    rmse |   mean_bias |   under_forecast_rate |   train_seconds |   inference_seconds |   n_features |   n_train |   n_val |   epochs | torch_version   |   cuda_available | device_name                        |
|:-------------------|:---------|---------:|---------:|--------:|------------:|----------------------:|----------------:|--------------------:|-------------:|----------:|--------:|---------:|:----------------|-----------------:|:-----------------------------------|
| train_mean_naive   | cpu      |     3780 | 1.09535  | 4.03351 |   -0.229743 |              0.34418  |          0      |             0       |          nan |       nan |     nan |      nan | nan             |              nan | nan                                |
| torch_temporal_mlp | cuda     |     3780 | 0.950702 | 4.09116 |   -1.28667  |              0.515608 |          2.0865 |             0.00098 |            3 |     17640 |    3780 |       15 | 2.11.0+cu128    |                1 | NVIDIA GeForce RTX 4090 Laptop GPU |

## Interpretation

The experiment verifies the GPU model-training path and produces comparable forecasting metrics against a simple train-mean baseline. This is a smoke experiment, not yet the final specialty method.

## Next step

Use the GPU model as a candidate forecast or residual-correction model, then evaluate it through the same bias, inventory-impact, A/B-style, and release-gate framework.