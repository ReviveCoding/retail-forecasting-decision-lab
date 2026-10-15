from __future__ import annotations
from collections.abc import Collection, Mapping
from typing import Any

import pandas as pd
from .bootstrap_ci import cluster_bootstrap_mean_ci
from .mde_power import approximate_mde


def _make_week_units(policy_rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = policy_rows.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week_start'] = df['date'].dt.to_period('W-MON').dt.start_time
    return df.groupby(['forecast_model', 'item_store_id', 'week_start'], as_index=False).agg(
        metric_value=(metric, 'mean')
    )


def build_ab_style_readout(
    detail: pd.DataFrame,
    control='tree_ensemble_policy',
    treatment='risk_safe_corrected_policy',
    lead_time=7,
    safety_stock_factor=0.5,
    n_bootstrap=200,
    seed=42,
    unit_type='item-store-week',
) -> pd.DataFrame:
    """Build an offline A/B-style readout using item-store-week units.

    The source data are not randomized experiments. This function makes an
    experiment-style policy comparison with clustered uncertainty, which is more
    conservative than day-row bootstrapping because repeated days for the same
    item-store are correlated.
    """
    subset = detail[(detail['lead_time'] == lead_time) & (detail['safety_stock_factor'] == safety_stock_factor)].copy()
    rows = []
    for metric in ['stockout_flag', 'service_level', 'total_cost', 'demand_coverage_f1', 'inventory_precision', 'overstock_units']:
        if subset.empty or metric not in subset.columns:
            continue
        weekly = _make_week_units(subset[subset['forecast_model'].isin([control, treatment])], metric)
        c = weekly[weekly['forecast_model'] == control]
        t = weekly[weekly['forecast_model'] == treatment]
        if c.empty or t.empty:
            continue
        merged = c[['item_store_id', 'week_start', 'metric_value']].merge(
            t[['item_store_id', 'week_start', 'metric_value']],
            on=['item_store_id', 'week_start'],
            suffixes=('_control', '_treatment'),
        )
        if merged.empty:
            continue
        merged['diff'] = merged['metric_value_treatment'] - merged['metric_value_control']
        c_val = float(merged['metric_value_control'].mean())
        t_val = float(merged['metric_value_treatment'].mean())
        delta = t_val - c_val
        ci_low, ci_high = cluster_bootstrap_mean_ci(merged, 'diff', 'item_store_id', n_bootstrap=n_bootstrap, seed=seed)
        rows.append({
            'experiment_id': 'offline_ab_policy_readout',
            'unit_type': unit_type,
            'control_policy': control,
            'treatment_policy': treatment,
            'metric_name': metric,
            'control_value': c_val,
            'treatment_value': t_val,
            'absolute_delta': float(delta),
            'relative_delta': float(delta / c_val) if c_val != 0 else 0.0,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'mde_approx': approximate_mde(merged['diff']),
            'n_units': int(len(merged)),
            'n_clusters': int(merged['item_store_id'].nunique()),
            'decision_hint': 'stable' if (ci_low > 0 or ci_high < 0) else 'uncertain',
        })
    return pd.DataFrame(rows)

def build_gpu_ab_style_readout(
    detail: pd.DataFrame,
    available_policies: Collection[str],
    experiment: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame | None:
    """Compare the highest-priority available GPU policy using the same experiment settings.

    Policy priority is explicit to keep comparisons stable across runs. In
    particular, use the configured lead time, stock factor, and bootstrap
    budget instead of relying on the default values of the base readout.
    """
    treatment = next(
        (
            policy
            for policy in (
                "learned_abc_release_safe_policy",
                "learned_abc_gpu_residual_policy",
                "release_safe_gpu_residual_policy",
                "guarded_gpu_residual_policy",
            )
            if policy in available_policies
        ),
        None,
    )
    if treatment is None:
        return None

    return build_ab_style_readout(
        detail,
        control="tree_ensemble_policy",
        treatment=treatment,
        lead_time=experiment.get("lead_time", 7),
        safety_stock_factor=experiment.get("safety_stock_factor", 0.5),
        n_bootstrap=experiment.get("n_bootstrap", 200),
        seed=experiment.get("random_seed", seed),
        unit_type=experiment.get("unit_type", "item-store-week"),
    )
