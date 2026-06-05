
from __future__ import annotations
import numpy as np
import pandas as pd


def add_demand_coverage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fulfilled = np.minimum(df['order_qty'].clip(lower=0), df['actual_demand'].clip(lower=0))
    demand_den = df['actual_demand'].replace(0, np.nan)
    inv_den = df['order_qty'].replace(0, np.nan)
    df['fulfilled_demand'] = fulfilled
    df['demand_recall'] = (fulfilled / demand_den).fillna(1.0)
    df['inventory_precision'] = (fulfilled / inv_den).fillna(1.0)
    denom = df['demand_recall'] + df['inventory_precision']
    df['demand_coverage_f1'] = (2 * df['demand_recall'] * df['inventory_precision'] / denom.replace(0, np.nan)).fillna(0)
    return df
