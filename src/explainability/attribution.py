
from __future__ import annotations
import pandas as pd


def top_features(feature_importance: pd.DataFrame, n=10) -> list[dict]:
    if feature_importance is None or feature_importance.empty:
        return []
    return feature_importance.head(n).to_dict('records')
