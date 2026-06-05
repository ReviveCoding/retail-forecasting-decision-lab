
from __future__ import annotations
import pandas as pd


def add_stockout_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['item_store_id','date']).copy()
    g = df.groupby('item_store_id', group_keys=False)
    df['zero_demand_recent_7'] = g['demand'].transform(lambda s: s.shift(1).fillna(0).eq(0).rolling(7, min_periods=1).mean())
    df['demand_drop_flag'] = (df['lag_7'].fillna(0) > 0).astype(int) * (df['rolling_mean_7'].fillna(0) < 0.5 * df['lag_7'].fillna(0)).astype(int)
    return df
