from __future__ import annotations


def model_card_text() -> str:
    return """
## Model family

Seasonal naive, moving-average, sklearn install-light tree-ensemble forecaster, and residual bias corrector.

## Intended use

Public-data retail demand forecasting benchmark and forecast-change measurement framework.

## Non-intended use

Not intended for production ordering, customer-facing automation, or claims about Amazon internal systems.

## Validation

Temporal holdout with final-window testing. Residual correction is trained on validation-window residuals only. Demand-State Timestep lifecycle age uses a fixed pre-declared reference horizon rather than future test-window length.
"""
