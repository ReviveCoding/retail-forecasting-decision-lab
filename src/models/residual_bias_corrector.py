
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .tree_forecaster import _make_X

@dataclass
class ResidualCorrector:
    model: RandomForestRegressor
    columns: list[str]


def train_residual_corrector(validation_df: pd.DataFrame, base_pred_col='pred_tree', seed=42) -> ResidualCorrector:
    df = validation_df.copy()
    df['residual_target'] = df['demand'] - df[base_pred_col]
    X = _make_X(df)
    y = df['residual_target'].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=2, random_state=int(seed), n_jobs=-1)
    model.fit(X, y)
    return ResidualCorrector(model=model, columns=list(X.columns))


def apply_residual_correction(corrector: ResidualCorrector, df: pd.DataFrame, base_pred_col='pred_tree') -> pd.Series:
    X = _make_X(df, columns=corrector.columns)
    residual_pred = corrector.model.predict(X)
    corrected = df[base_pred_col].values + residual_pred
    return pd.Series(np.maximum(corrected, 0), index=df.index)
