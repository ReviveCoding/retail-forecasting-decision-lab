# Executive Summary

Retail Forecasting Lab completed a sample-mode run on M5 public retail demand data.

## Decision

**BLOCK**

## What changed in v3

- Consolidated configuration into `configs/pipeline.yaml`.
- Separated point-forecast metrics from risk-safe inventory policy metrics.
- Fixed release-gate overstock and total-cost guardrails so they can actually fail.
- Replaced row-level A/B readout with item-store-week units and clustered bootstrap intervals.
- Moved optional non-core dataset adapters into `extensions/`.
- Wrote lightweight detail samples by default instead of large detail CSVs.
- Added interpretation-oriented reports instead of table-only report dumps.

## Core outputs

- Forecast metrics: `outputs/forecast_metrics.csv`
- Bias by slice: `outputs/bias_by_slice.csv`
- Inventory policy summary: `outputs/inventory_policy_summary.csv`
- Offline A/B-style readout: `outputs/ab_test_summary.csv`
- Release gate: `outputs/release_gate.json`

## Claim boundary

This is a public-data, local-runnable, offline forecasting measurement framework. Inventory and lead-time impacts use synthetic/proxy overlays.
