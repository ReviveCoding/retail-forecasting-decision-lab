from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metric_row(df: pd.DataFrame, pred_col: str, model_name: str) -> dict:
    """Evaluate demand forecasts and report signed bias as forecast minus observed demand."""
    observed_demand = df["demand"].astype(float).values
    predicted_demand = df[pred_col].astype(float).clip(lower=0).values
    forecast_error = predicted_demand - observed_demand
    absolute_error = np.abs(forecast_error)
    return {
        "model_name": model_name,
        "n_obs": int(len(df)),
        "rmse": float(np.sqrt(np.mean(forecast_error**2))) if len(df) else 0.0,
        "mae": float(np.mean(absolute_error)) if len(df) else 0.0,
        "wape": float(absolute_error.sum() / max(np.abs(observed_demand).sum(), 1e-9)),
        "mean_bias": float(np.mean(forecast_error)) if len(df) else 0.0,
        "under_forecast_rate": float(np.mean(predicted_demand < observed_demand))
        if len(df)
        else 0.0,
        "over_forecast_rate": float(np.mean(predicted_demand > observed_demand))
        if len(df)
        else 0.0,
    }


def evaluate_forecasts(df: pd.DataFrame, pred_map: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            forecast_metric_row(df, col, name)
            for name, col in pred_map.items()
            if col in df.columns
        ]
    )
