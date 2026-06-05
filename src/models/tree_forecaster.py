from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DEFAULT_FEATURES = [
    'lag_7','lag_14','lag_28','rolling_mean_7','rolling_std_7','rolling_mean_28','rolling_std_28',
    'sell_price','price_change_7','event_flag','snap_flag','dayofweek','month','weekofyear','is_weekend',
    'cat_demand_lag_mean','dept_demand_lag_mean','store_demand_lag_mean','state_demand_lag_mean',
    'days_since_series_start','normalized_lifecycle_age','rolling_demand_state','volatility_state',
    'intermittency_score','event_recent_7','category_relative_age','zero_demand_recent_7','demand_drop_flag'
]
CATEGORICAL_FEATURES = ['cat_id','dept_id','store_id','state_id']

@dataclass
class ForecastModel:
    model: Any
    columns: list[str]
    feature_importance: pd.DataFrame
    algorithm: str = 'RandomForestRegressor'


def _make_X(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    X = df[numeric_cols + [c for c in CATEGORICAL_FEATURES if c in df.columns]].copy()
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def train_forecaster(
    train_df: pd.DataFrame,
    seed: int = 42,
    n_estimators: int = 80,
    max_depth: int | None = 12,
    min_samples_leaf: int = 2,
    **_: Any,
) -> ForecastModel:
    """Train an install-light tree-ensemble forecasting baseline.

    The default is intentionally Windows-friendly and avoids optional compiled
    dependencies. The repository therefore calls this a tree-ensemble forecaster,
    not a production LightGBM/XGBoost implementation.
    """
    X = _make_X(train_df)
    y = train_df['demand'].astype(float).clip(lower=0)
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if max_depth else None,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X, y)
    imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return ForecastModel(model=model, columns=list(X.columns), feature_importance=imp)


def predict_forecaster(forecast_model: ForecastModel, df: pd.DataFrame) -> pd.Series:
    X = _make_X(df, columns=forecast_model.columns)
    pred = forecast_model.model.predict(X)
    return pd.Series(pred, index=df.index).clip(lower=0)
