
from __future__ import annotations
import pandas as pd


def add_calendar_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['price_lag_7'] = df.groupby('item_store_id')['sell_price'].shift(7)
    df['price_change_7'] = (df['sell_price'] - df['price_lag_7']) / df['price_lag_7'].replace(0, pd.NA)
    df['price_change_7'] = df['price_change_7'].replace([float('inf'), -float('inf')], 0).fillna(0)
    return df
