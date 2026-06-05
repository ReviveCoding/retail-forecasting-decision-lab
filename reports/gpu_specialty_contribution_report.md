# GPU Specialty Residual Correction Report

## Purpose

This report consolidates the optional CUDA-backed PyTorch residual-correction experiment. The core forecasting pipeline remains CPU-runnable, while this GPU extension evaluates whether a learned residual corrector can improve the tree-ensemble baseline.

## GPU validation

- CUDA available: `True`
- PyTorch version: `2.11.0+cu128`
- PyTorch CUDA runtime: `12.8`
- Device: `NVIDIA GeForce RTX 4090 Laptop GPU`

## Consolidated metrics

| method                         |     wape |    rmse |    mean_bias |   under_forecast_rate |   relative_wape_improvement_vs_tree_pct | notes                                                                                    |   lambda |   clip_quantile |
|:-------------------------------|---------:|--------:|-------------:|----------------------:|----------------------------------------:|:-----------------------------------------------------------------------------------------|---------:|----------------:|
| tree_ensemble_baseline         | 0.68948  | 3.02234 |  0.209212    |              0.345238 |                                0        | CPU tree-ensemble reference forecast.                                                    |    nan   |          nan    |
| current_residual_corrected     | 0.694348 | 3.10019 | -0.000878894 |              0.357143 |                               -0.706018 | Existing CPU residual correction artifact.                                               |    nan   |          nan    |
| raw_gpu_residual_corrected     | 0.651888 | 3.07841 | -0.792444    |              0.496032 |                                5.45226  | CUDA-backed PyTorch residual corrector before guardrails.                                |    nan   |          nan    |
| guarded_gpu_residual_corrected | 0.670724 | 3.00561 |  0.00122396  |              0.400794 |                                2.72037  | Guarded GPU residual correction using residual clipping and correction-strength scaling. |      0.2 |            0.95 |

## Interpretation

- Raw GPU residual correction improved WAPE from `0.689480` to `0.651888`, a `5.45%` relative improvement versus the tree baseline.
- Guarded GPU residual correction selected `guarded_torch_residual_lambda_0.2_q_0.95` and preserved WAPE improvement at `0.670724`, a `2.72%` relative improvement versus the tree baseline.
- The guarded version reduced mean bias from `0.209212` for the tree baseline to `0.001224`, while avoiding the stronger negative-bias shift of raw GPU correction.

## Claim boundary

This is an optional GPU specialty experiment built on generated forecast artifacts. It is useful as GPU ML feasibility evidence and specialty-method evidence. Production-style or resume-critical claims should be made after integrating the guarded GPU residual corrector directly into the main leakage-safe train/validation/test pipeline.