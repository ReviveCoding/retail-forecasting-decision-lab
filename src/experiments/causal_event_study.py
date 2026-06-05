
from __future__ import annotations
import numpy as np
import pandas as pd
from .bootstrap_ci import bootstrap_mean_ci


def lightweight_event_residual_readout(test_df: pd.DataFrame, pred_col='pred_tree', n_bootstrap=200, seed=42) -> pd.DataFrame:
    df = test_df.copy()
    df['residual'] = df['demand'] - df[pred_col]
    if df['event_flag'].nunique() < 2:
        df['price_change_treated'] = (df['price_change_7'].abs() > df['price_change_7'].abs().median()).astype(int)
        treatment_col = 'price_change_treated'
    else:
        treatment_col = 'event_flag'
    treated = df[df[treatment_col] == 1]['residual']
    control = df[df[treatment_col] == 0]['residual']
    if len(treated) == 0 or len(control) == 0:
        return pd.DataFrame([{'analysis':'observational_event_readout','treatment':treatment_col,'estimate':0.0,'ci_low':0.0,'ci_high':0.0,'note':'insufficient treated/control observations'}])
    estimate = treated.mean() - control.mean()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_bootstrap)):
        boot.append(rng.choice(treated, len(treated), replace=True).mean() - rng.choice(control, len(control), replace=True).mean())
    return pd.DataFrame([{
        'analysis':'observational_event_readout',
        'treatment': treatment_col,
        'outcome': 'forecast_residual_actual_minus_prediction',
        'treated_n': int(len(treated)),
        'control_n': int(len(control)),
        'estimate': float(estimate),
        'ci_low': float(np.quantile(boot, 0.025)),
        'ci_high': float(np.quantile(boot, 0.975)),
        'note': 'observational readout only; not randomized causal proof',
    }])
