from __future__ import annotations
import numpy as np
import pandas as pd


def add_demand_state_timestep(df: pd.DataFrame, lifecycle_reference_days: int = 365) -> pd.DataFrame:
    """PHT-inspired statistical Demand-State Timestep v1.

    The feature represents lifecycle/state information using only information
    available up to the current timestamp.  Earlier versions normalized lifecycle
    age by the full item-store series length, which is convenient but can look
    leakage-like in a temporal holdout.  This implementation uses a fixed
    pre-declared reference horizon, not future test-window length.
    """
    df = df.sort_values(['item_store_id','date']).copy()
    g = df.groupby('item_store_id', group_keys=False)
    df['days_since_series_start'] = g.cumcount()
    ref = max(int(lifecycle_reference_days), 1)
    df['normalized_lifecycle_age'] = (df['days_since_series_start'] / ref).clip(lower=0, upper=1)
    hist_mean = g['demand'].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    hist_std = g['demand'].transform(lambda s: s.shift(1).rolling(28, min_periods=2).std())
    df['rolling_demand_state'] = hist_mean.fillna(0)
    df['volatility_state'] = (hist_std / hist_mean.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    df['intermittency_score'] = g['demand'].transform(lambda s: (s.shift(1).fillna(0).eq(0)).rolling(28, min_periods=1).mean())
    df['event_recent_7'] = g['event_flag'].transform(lambda s: s.shift(1).fillna(0).rolling(7, min_periods=1).max())
    cat_age = df.groupby(['cat_id','date'])['normalized_lifecycle_age'].transform('mean')
    df['category_relative_age'] = df['normalized_lifecycle_age'] - cat_age
    return df
