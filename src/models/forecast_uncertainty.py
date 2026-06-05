
from __future__ import annotations
import pandas as pd


def add_candidate_forecast_uncertainty(df: pd.DataFrame, pred_cols=None) -> pd.DataFrame:
    df = df.copy()
    pred_cols = pred_cols or ['pred_seasonal_naive','pred_moving_average','pred_tree','pred_corrected']
    cols = [c for c in pred_cols if c in df.columns]
    df['candidate_forecast_mean'] = df[cols].mean(axis=1)
    df['candidate_forecast_std'] = df[cols].std(axis=1).fillna(0)
    df['model_disagreement_score'] = df['candidate_forecast_std'] / df['candidate_forecast_mean'].replace(0, 1)
    df['pred_p10'] = (df['candidate_forecast_mean'] - 1.28 * df['candidate_forecast_std']).clip(lower=0)
    df['pred_p90'] = (df['candidate_forecast_mean'] + 1.28 * df['candidate_forecast_std']).clip(lower=0)
    return df
