# Observational Causal Driver Analysis

## What was tested

Estimated residual shifts around event or price-change indicators on the holdout set.

## Baseline

Correlation or feature importance alone does not establish causal behavior.

## Enhanced method

The readout separates treated/control windows and reports uncertainty.

## Key results

| analysis                    | treatment   | outcome                                   |   treated_n |   control_n |   estimate |    ci_low |   ci_high | note                                                    |
|:----------------------------|:------------|:------------------------------------------|------------:|------------:|-----------:|----------:|----------:|:--------------------------------------------------------|
| observational_event_readout | event_flag  | forecast_residual_actual_minus_prediction |         240 |        1440 |  0.0878489 | -0.251336 |  0.516872 | observational readout only; not randomized causal proof |

## Interpretation

This is a lightweight observational diagnostic suitable for MVP; Favorita promotion/IPW/pretrend checks are the next extension.

## Claim boundary

No randomized causal proof is claimed.
