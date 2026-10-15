# Retail Forecasting Lab

**Demand Forecasting, Bias Correction, Business Impact Measurement & Forecast Explainability Platform**

This repository is a local-runnable, production-style data science framework for retail demand forecasting. It is designed around five integrated subprojects:

1. Public-data demand forecasting core
2. Demand-State Timestep and residual forecast bias correction
3. Inventory, lead-time, and business-impact simulation
4. Offline A/B-style measurement and lightweight causal readout
5. Forecast explanation grounding evaluator

## Claim boundary

This project uses public retail forecasting data for demand forecasting, temporal holdout evaluation, price/calendar analysis, and residual bias correction. Inventory, stockout, lead-time, service-level, and labor-planning impacts are evaluated through synthetic or proxy overlays because the public datasets do not provide Amazon internal inventory, replenishment, or warehouse-labor data.

This is **not** an Amazon production system, does **not** use Amazon internal data, and does **not** claim production A/B testing. It is a local-runnable public-data benchmark and measurement framework.

## Core dataset

The core pipeline requires the M5 Forecasting Accuracy data as a ZIP archive or extracted directory. The default configuration looks for `data/raw/m5-forecasting-accuracy.zip` relative to the repository root. Set the `M5_DATA_PATH` environment variable to use a different ZIP or extracted directory without editing the tracked configuration. You can also supply a custom `--config` YAML file. Dataset files are not bundled with this repository.

## Model and evaluation updates

This version removes the main structural inefficiencies from the earlier MVP:

- Configuration is consolidated into `configs/pipeline.yaml`.
- Forecast metrics and inventory-policy metrics are separated. The risk-safe correction is an ordering-policy signal, not a point forecast.
- Release-gate overstock and total-cost guardrails now actually fail when thresholds are violated.
- Offline A/B-style measurement uses item-store-week units and clustered bootstrap intervals instead of row-level day bootstrapping.
- Optional Favorita/Instacart adapters are moved to `extensions/` so the core pipeline remains focused on M5.
- Large detail outputs are sample-only by default; full detail output is controlled by config.
- Reports include interpretation sections, not just table dumps.

## What this repo demonstrates

| Subproject | Capability shown | Baseline improvement goal |
|---|---|---|
| Public-data forecasting core | Time-series split, features, baselines, install-light tree-ensemble forecaster | Improve WAPE/RMSE and slice diagnostics over seasonal/moving-average baselines |
| Demand-State Timestep + bias correction | Data-centric lifecycle features and residual correction | Reduce systematic under-forecasting and worst-slice bias |
| Business-impact simulator | Forecast-to-inventory decision metrics | Translate forecast changes into stockout, service level, overstock, cost, and coverage F1 |
| Offline A/B-style measurement | Treatment/control readout, clustered bootstrap CI, MDE | Evaluate whether corrected forecast policy is launchable under guardrails |
| Explanation grounding | Forecast explanation reliability checks | Detect unsupported or numerically inconsistent explanations |

## Quick start

```bash
cd retail_forecasting_lab
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_pipeline --config configs/pipeline.yaml
python -m pytest -q
```

The default run uses sample mode for fast local execution. Edit `configs/pipeline.yaml` to change sample size. Start with sample mode on a local machine; full mode converts the wide M5 sales table to an item-store-day panel and may require substantially more memory.

## Main outputs

```text
outputs/forecast_metrics.csv
outputs/bias_by_slice.csv
outputs/corrected_forecasts_sample.csv
outputs/inventory_policy_summary.csv
outputs/inventory_policy_detail_sample.csv
outputs/demand_coverage_metrics_sample.csv
outputs/ab_test_summary.csv
outputs/causal_effect_estimates.csv
outputs/explanation_eval_metrics.json
outputs/release_gate.json

reports/executive_summary.md
reports/m5_forecast_report.md
reports/bias_correction_report.md
reports/inventory_policy_report.md
reports/ab_style_experiment_readout.md
reports/causal_driver_analysis.md
reports/forecast_explainability_report.md
reports/launch_decision_memo.md
reports/leakage_audit.md
reports/failure_case_gallery.md
```

## Pipeline

```text
M5 raw data
→ schema validation
→ item-store-date panel
→ lag/rolling/calendar/price/hierarchy features
→ seasonal naive / moving average / install-light tree-ensemble forecaster
→ Demand-State Timestep features
→ residual bias correction
→ forecast/bias/slice evaluation
→ synthetic inventory and lead-time simulation
→ Demand Recall / Inventory Precision / Demand Coverage F1
→ offline A/B-style measurement with item-store-week units
→ lightweight observational event readout
→ forecast explanation grounding
→ release gate and stakeholder memo
```

## Leakage-safety rules

- Lag and rolling features use only information available before the prediction date.
- Residual correction is trained on validation-window residuals and evaluated on the final test window.
- Demand-State Timestep v1 uses historical lifecycle and volatility information. Lifecycle age is normalized against a fixed pre-declared reference horizon rather than the full future series length.
- The risk-safe corrected signal is evaluated as an inventory policy, not as a point forecast in WAPE metrics.
- Inventory orders are computed from forecasts, safety-stock assumptions, and historical uncertainty, not from actual test demand.
- Causal readout is labeled observational and does not claim randomized causal proof.

## Optional extensions

Favorita and Instacart adapters live under `extensions/optional_data_adapters/`. They are intentionally not imported by the core M5 pipeline until their schemas are mapped into the canonical panel.

## Optional GPU residual-correction extension

The core pipeline remains CPU-runnable. An optional CUDA-backed PyTorch extension validates GPU training and inference on an NVIDIA RTX 4090 Laptop GPU and evaluates a residual-correction model on generated forecast artifacts.

Run:

```powershell
python -m pip install -r requirements-gpu.txt
python scripts\run_gpu_residual_experiment.py
```

Current exploratory GPU result:

| Method | WAPE | RMSE | Mean Bias | Under-Forecast Rate | Relative WAPE Improvement vs Tree |
|---|---:|---:|---:|---:|---:|
| Tree ensemble baseline | 0.6895 | 3.0223 | 0.2092 | 0.3452 | 0.00% |
| Raw GPU residual corrected | 0.6519 | 3.0784 | -0.7924 | 0.4960 | 5.45% |
| Guarded GPU residual corrected | 0.6707 | 3.0056 | 0.0012 | 0.4008 | 2.72% |

The raw GPU residual corrector maximized WAPE improvement but shifted bias negatively. The guarded variant used residual clipping and correction-strength scaling to retain WAPE improvement while stabilizing mean bias.

Claim boundary: this is an optional GPU specialty experiment built on generated forecast artifacts. Production-style claims should come from integrating the guarded GPU residual corrector directly into the main leakage-safe train/validation/test pipeline.

## Model selection frontier

The framework separates the accuracy frontier from the inventory-policy frontier. In the current run, `learned_abc_gpu_residual` is the accuracy-optimal forecast, while `learned_abc_release_safe_policy` is the release-gate treatment policy.

Key outputs:

- `outputs/model_selection_frontier.csv`
- `outputs/model_selection_recommendation.json`
- `reports/model_selection_frontier_report.md`

Current decision: `ITERATE`. The learned ABC forecast improves WAPE versus the tree baseline, and the release-safe policy passes stockout, overstock, total-cost, and worst-slice guardrails. The remaining iteration item is the under-forecast guardrail.

Claim boundary: this is an offline public-data decision artifact. Inventory impacts use synthetic/proxy overlays.