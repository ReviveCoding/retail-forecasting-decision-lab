# Bias Correction Report

## What was tested

Measured whether validation-trained residual correction changes forecast bias on the test window.

## Baseline

The control forecast is the tree-ensemble point forecast.

## Enhanced method

The corrected forecast adds a learned residual estimate to the control forecast.

## Key results

## Forecast metrics

| model_name                |   n_obs |    rmse |     mae |     wape |   mean_bias |   under_forecast_rate |   over_forecast_rate |
|:--------------------------|--------:|--------:|--------:|---------:|------------:|----------------------:|---------------------:|
| seasonal_naive            |    1680 | 3.51316 | 2.00893 | 0.90024  |  -0.250595  |              0.382143 |             0.297619 |
| moving_average            |    1680 | 2.86722 | 1.66671 | 0.746885 |  -0.109141  |              0.41369  |             0.576786 |
| tree_ensemble             |    1680 | 2.81444 | 1.74862 | 0.78359  |   0.38278   |              0.3375   |             0.6625   |
| corrected                 |    1680 | 2.75804 | 1.69569 | 0.759871 |   0.190909  |              0.356548 |             0.627381 |
| guarded_gpu_residual      |    1680 | 2.8008  | 1.63797 | 0.734004 |  -0.0810581 |              0.414881 |             0.578571 |
| release_safe_gpu_residual |    1680 | 2.84307 | 1.77195 | 0.794044 |   0.445909  |              0.327381 |             0.672619 |

## Slice metrics

| model_name                |   n_obs |    rmse |     mae |     wape |   mean_bias |   under_forecast_rate |   over_forecast_rate | slice_type   | slice_value   | worst_slice_flag   |
|:--------------------------|--------:|--------:|--------:|---------:|------------:|----------------------:|---------------------:|:-------------|:--------------|:-------------------|
| seasonal_naive            |    1680 | 3.51316 | 2.00893 | 0.90024  |  -0.250595  |              0.382143 |             0.297619 | cat_id       | FOODS         | True               |
| moving_average            |    1680 | 2.86722 | 1.66671 | 0.746885 |  -0.109141  |              0.41369  |             0.576786 | cat_id       | FOODS         | True               |
| tree_ensemble             |    1680 | 2.81444 | 1.74862 | 0.78359  |   0.38278   |              0.3375   |             0.6625   | cat_id       | FOODS         | True               |
| corrected                 |    1680 | 2.75804 | 1.69569 | 0.759871 |   0.190909  |              0.356548 |             0.627381 | cat_id       | FOODS         | True               |
| guarded_gpu_residual      |    1680 | 2.8008  | 1.63797 | 0.734004 |  -0.0810581 |              0.414881 |             0.578571 | cat_id       | FOODS         | True               |
| release_safe_gpu_residual |    1680 | 2.84307 | 1.77195 | 0.794044 |   0.445909  |              0.327381 |             0.672619 | cat_id       | FOODS         | True               |
| seasonal_naive            |     840 | 2.9417  | 1.6631  | 1.01159  |  -0.272619  |              0.382143 |             0.280952 | store_id     | CA_1          | True               |
| moving_average            |     840 | 2.33835 | 1.39179 | 0.846566 |  -0.125638  |              0.40119  |             0.589286 | store_id     | CA_1          | True               |
| tree_ensemble             |     840 | 2.37364 | 1.52888 | 0.929951 |   0.396263  |              0.314286 |             0.685714 | store_id     | CA_1          | True               |
| corrected                 |     840 | 2.31526 | 1.45362 | 0.88417  |   0.224199  |              0.353571 |             0.62619  | store_id     | CA_1          | True               |

## Interpretation

Correction should be judged by both accuracy and under-forecast risk. Mixed results are routed through the release gate instead of being overclaimed.

## Claim boundary

Residual correction is trained on validation-window residuals only.
