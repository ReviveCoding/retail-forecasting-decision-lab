# M5 Forecast Report

## What was tested

Compared seasonal naive, moving-average, tree-ensemble, and residual-corrected forecasts on a temporal holdout.

## Baseline

Seasonal naive and moving-average forecasts represent install-light vanilla demand baselines.

## Enhanced method

The enhanced path adds lag/rolling/calendar/price/hierarchy features and residual correction.

## Key results

| model_name                |   n_obs |    rmse |     mae |     wape |   mean_bias |   under_forecast_rate |   over_forecast_rate |
|:--------------------------|--------:|--------:|--------:|---------:|------------:|----------------------:|---------------------:|
| seasonal_naive            |    1680 | 3.51316 | 2.00893 | 0.90024  |  -0.250595  |              0.382143 |             0.297619 |
| moving_average            |    1680 | 2.86722 | 1.66671 | 0.746885 |  -0.109141  |              0.41369  |             0.576786 |
| tree_ensemble             |    1680 | 2.81444 | 1.74862 | 0.78359  |   0.38278   |              0.3375   |             0.6625   |
| corrected                 |    1680 | 2.75624 | 1.69372 | 0.758991 |   0.186047  |              0.357738 |             0.62619  |
| guarded_gpu_residual      |    1680 | 2.8008  | 1.63797 | 0.734004 |  -0.0810581 |              0.414881 |             0.578571 |
| release_safe_gpu_residual |    1680 | 2.84307 | 1.77195 | 0.794044 |   0.445909  |              0.327381 |             0.672619 |
| learned_abc_gpu_residual  |    1680 | 2.75485 | 1.61314 | 0.722877 |  -0.156777  |              0.426786 |             0.560119 |

## Interpretation

Use WAPE/RMSE for point-forecast accuracy and under-forecast rate for operational risk.

## Claim boundary

Results are sample-mode public-data benchmark results, not production Amazon metrics.
