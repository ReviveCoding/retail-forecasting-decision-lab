
from __future__ import annotations
import pandas as pd


def add_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pred_seasonal_naive'] = df['lag_28'].fillna(df['lag_7']).fillna(df['rolling_mean_7']).fillna(0)
    df['pred_moving_average'] = df['rolling_mean_28'].fillna(df['rolling_mean_7']).fillna(df['pred_seasonal_naive']).fillna(0)
    return df
