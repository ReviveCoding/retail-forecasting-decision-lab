# Leakage-Safe Guarded GPU Residual Corrector

## Purpose

This integrated stage trains a CUDA-backed PyTorch residual model without using the test window for fitting, scaling, early stopping target construction, or guard selection.

## Split design

- train_base rows: 16127
- residual_fit rows: 4033
- guard_val rows: 1680
- test rows: 1680

## Test metrics

| model                          | device   |   n_test |     wape |    rmse |   mean_bias |   under_forecast_rate |   train_seconds |   inference_seconds |
|:-------------------------------|:---------|---------:|---------:|--------:|------------:|----------------------:|----------------:|--------------------:|
| gpu_base_tree                  | cpu      |     1680 | 0.754965 | 2.80524 |   0.141518  |              0.357738 |          0      |            0        |
| raw_gpu_residual_corrected     | cuda     |     1680 | 0.743067 | 3.04435 |  -1.11039   |              0.520238 |          1.5357 |            0.001153 |
| guarded_gpu_residual_corrected | cuda     |     1680 | 0.734004 | 2.8008  |  -0.0810581 |              0.414881 |          1.5357 |            0.001153 |

## Selected guard

| model                         | device   |   n_test |     wape |    rmse |   mean_bias |   under_forecast_rate |   train_seconds |   inference_seconds |   lambda |   clip_quantile |   clip_low |   clip_high |   score |
|:------------------------------|:---------|---------:|---------:|--------:|------------:|----------------------:|----------------:|--------------------:|---------:|----------------:|-----------:|------------:|--------:|
| guarded_gpu_lambda_0.2_q_0.95 | cuda     |     1680 | 0.719899 | 2.55073 | -0.00325214 |              0.356548 |               0 |                   0 |      0.2 |            0.95 |   -1.32955 |    -0.05795 |      22 |

## Claim boundary

The residual model is fit on residual_fit only. Lambda and clipping are selected on guard_val only. Test is used only once for final evaluation.