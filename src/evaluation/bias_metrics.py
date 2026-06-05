
from __future__ import annotations
import pandas as pd
from .forecast_metrics import forecast_metric_row


def evaluate_bias_by_slice(df: pd.DataFrame, pred_map: dict[str,str], slice_cols=('cat_id','store_id','state_id','event_flag')) -> pd.DataFrame:
    rows = []
    for slice_col in slice_cols:
        if slice_col not in df.columns:
            continue
        for value, g in df.groupby(slice_col):
            for model_name, pred_col in pred_map.items():
                if pred_col in g.columns:
                    row = forecast_metric_row(g, pred_col, model_name)
                    row.update({'slice_type': slice_col, 'slice_value': str(value)})
                    rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out['worst_slice_flag'] = out.groupby(['model_name','slice_type'])['wape'].transform(lambda s: s == s.max())
    return out
