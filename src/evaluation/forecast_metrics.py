
from __future__ import annotations
import numpy as np
import pandas as pd


def forecast_metric_row(df: pd.DataFrame, pred_col: str, model_name: str) -> dict:
    y = df['demand'].astype(float).values
    p = df[pred_col].astype(float).clip(lower=0).values
    err = p - y
    abs_err = np.abs(err)
    return {
        'model_name': model_name,
        'n_obs': int(len(df)),
        'rmse': float(np.sqrt(np.mean(err ** 2))) if len(df) else 0.0,
        'mae': float(np.mean(abs_err)) if len(df) else 0.0,
        'wape': float(abs_err.sum() / max(np.abs(y).sum(), 1e-9)),
        'mean_bias': float(np.mean(err)) if len(df) else 0.0,
        'under_forecast_rate': float(np.mean(p < y)) if len(df) else 0.0,
        'over_forecast_rate': float(np.mean(p > y)) if len(df) else 0.0,
    }


def evaluate_forecasts(df: pd.DataFrame, pred_map: dict[str,str]) -> pd.DataFrame:
    return pd.DataFrame([forecast_metric_row(df, col, name) for name, col in pred_map.items() if col in df.columns])
