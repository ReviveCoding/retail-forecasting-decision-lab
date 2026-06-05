
from __future__ import annotations
import pandas as pd


def add_hierarchy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['date','item_store_id']).copy()
    # Use historical expanding means shifted by one day within hierarchy to avoid using same-day target.
    for key, name in [(['cat_id'], 'cat'), (['dept_id'], 'dept'), (['store_id'], 'store'), (['state_id'], 'state')]:
        daily = df.groupby(key + ['date'], as_index=False)['demand'].sum().sort_values(key + ['date'])
        daily[f'{name}_demand_lag_mean'] = daily.groupby(key)['demand'].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        df = df.merge(daily[key + ['date', f'{name}_demand_lag_mean']], on=key + ['date'], how='left')
    return df
