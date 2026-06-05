
from __future__ import annotations
import pandas as pd


def add_lag_rolling_features(df: pd.DataFrame, lags=(7,14,28), windows=(7,28)) -> pd.DataFrame:
    df = df.sort_values(['item_store_id','date']).copy()
    g = df.groupby('item_store_id', group_keys=False)
    for lag in lags:
        df[f'lag_{lag}'] = g['demand'].shift(lag)
    for w in windows:
        df[f'rolling_mean_{w}'] = g['demand'].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f'rolling_std_{w}'] = g['demand'].transform(lambda s: s.shift(1).rolling(w, min_periods=2).std())
    return df
