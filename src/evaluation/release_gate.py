from __future__ import annotations
from math import isfinite

import pandas as pd


def _metric(metrics_df, model, name):
    if metrics_df is None or metrics_df.empty or 'model_name' not in metrics_df.columns:
        return None
    row = metrics_df[metrics_df['model_name'] == model]
    if row.empty or name not in row.columns:
        return None
    return float(row.iloc[0][name])


def _safe_rel_improvement(control_value, treatment_value, lower_is_better=True):
    if control_value is None or treatment_value is None or abs(control_value) < 1e-12:
        return 0.0
    if lower_is_better:
        return (control_value - treatment_value) / abs(control_value)
    return (treatment_value - control_value) / abs(control_value)


def _policy_metric(inventory_summary, policy, metric, lead_time=7, safety_stock_factor=0.5):
    required = {'lead_time', 'safety_stock_factor', 'forecast_model', metric}
    if inventory_summary is None or inventory_summary.empty or not required.issubset(inventory_summary.columns):
        return None
    inv = inventory_summary[
        (inventory_summary['lead_time'] == lead_time) &
        (inventory_summary['safety_stock_factor'] == safety_stock_factor)
    ]
    row = inv[inv['forecast_model'] == policy]
    if row.empty or metric not in row.columns:
        return None
    return float(row.iloc[0][metric])


def _worst_slice_wape(bias_by_slice, model):
    if bias_by_slice is None or bias_by_slice.empty or 'wape' not in bias_by_slice.columns:
        return None
    row = bias_by_slice[bias_by_slice['model_name'] == model]
    if row.empty:
        return None
    return float(row['wape'].max())


def _ab_ci_stable(ab_summary, metric_name='stockout_flag', expected_direction='decrease'):
    if ab_summary is None or ab_summary.empty:
        return False
    row = ab_summary[ab_summary['metric_name'] == metric_name]
    if row.empty:
        return False
    low, high = float(row.iloc[0]['ci_low']), float(row.iloc[0]['ci_high'])
    if expected_direction == 'decrease':
        return high < 0
    return low > 0


def build_release_gate(
    metrics_df: pd.DataFrame,
    inventory_summary: pd.DataFrame,
    ab_summary: pd.DataFrame,
    bias_by_slice: pd.DataFrame | None = None,
    control_forecast='tree_ensemble',
    treatment_forecast='corrected',
    control_policy='tree_ensemble_policy',
    treatment_policy='risk_safe_corrected_policy',
    lead_time=7,
    safety_stock_factor=0.5,
    thresholds: dict | None = None,
) -> dict:
    thresholds = thresholds or {}
    min_wape = float(thresholds.get('min_wape_relative_improvement', 0.01))
    min_under = float(thresholds.get('min_under_forecast_relative_improvement', 0.03))
    max_overstock_worse = float(thresholds.get('max_overstock_relative_worsening', 0.02))
    max_cost_worse = float(thresholds.get('max_total_cost_relative_worsening', 0.02))
    max_worst_slice_worse = float(thresholds.get('max_worst_slice_wape_relative_worsening', 0.02))
    require_ci = bool(thresholds.get('require_ci_sign_stable_for_launch', False))

    control_wape = _metric(metrics_df, control_forecast, 'wape')
    treatment_wape = _metric(metrics_df, treatment_forecast, 'wape')
    control_under = _metric(metrics_df, control_forecast, 'under_forecast_rate')
    treatment_under = _metric(metrics_df, treatment_forecast, 'under_forecast_rate')
    wape_rel_improvement = _safe_rel_improvement(control_wape, treatment_wape, lower_is_better=True)
    under_rel_improvement = _safe_rel_improvement(control_under, treatment_under, lower_is_better=True)

    c_stockout = _policy_metric(inventory_summary, control_policy, 'stockout_rate', lead_time, safety_stock_factor)
    t_stockout = _policy_metric(inventory_summary, treatment_policy, 'stockout_rate', lead_time, safety_stock_factor)
    c_overstock = _policy_metric(inventory_summary, control_policy, 'overstock_units', lead_time, safety_stock_factor)
    t_overstock = _policy_metric(inventory_summary, treatment_policy, 'overstock_units', lead_time, safety_stock_factor)
    c_cost = _policy_metric(inventory_summary, control_policy, 'total_cost', lead_time, safety_stock_factor)
    t_cost = _policy_metric(inventory_summary, treatment_policy, 'total_cost', lead_time, safety_stock_factor)

    stockout_delta = None if c_stockout is None or t_stockout is None else t_stockout - c_stockout
    overstock_rel_worsening = None if c_overstock is None or t_overstock is None else (t_overstock - c_overstock) / max(abs(c_overstock), 1e-9)
    total_cost_rel_worsening = None if c_cost is None or t_cost is None else (t_cost - c_cost) / max(abs(c_cost), 1e-9)

    c_worst = _worst_slice_wape(bias_by_slice, control_forecast)
    t_worst = _worst_slice_wape(bias_by_slice, treatment_forecast)
    worst_slice_rel_worsening = None if c_worst is None or t_worst is None else (t_worst - c_worst) / max(abs(c_worst), 1e-9)
    ci_stable = _ab_ci_stable(ab_summary, metric_name='stockout_flag', expected_direction='decrease')

    guardrails = {
        'wape_improved': wape_rel_improvement >= 0.0,
        'under_forecast_improved': under_rel_improvement >= 0.0,
        'stockout_not_worse': (stockout_delta is None) or (stockout_delta <= 0.0),
        'overstock_guardrail_observed': (overstock_rel_worsening is None) or (overstock_rel_worsening <= max_overstock_worse),
        'total_cost_guardrail_observed': (total_cost_rel_worsening is None) or (total_cost_rel_worsening <= max_cost_worse),
        'worst_slice_wape_guardrail_observed': (worst_slice_rel_worsening is None) or (worst_slice_rel_worsening <= max_worst_slice_worse),
        'ab_ci_stable_if_required': (not require_ci) or ci_stable,
    }

    required_values = (
        control_wape, treatment_wape, control_under, treatment_under,
        c_stockout, t_stockout, c_overstock, t_overstock, c_cost, t_cost,
    )
    evidence_complete = all(value is not None and isfinite(value) for value in required_values)

    if not evidence_complete:
        decision = 'HOLD'
    elif all(guardrails.values()) and wape_rel_improvement >= min_wape and under_rel_improvement >= min_under:
        decision = 'LAUNCH'
    elif not guardrails['stockout_not_worse'] or not guardrails['overstock_guardrail_observed'] or not guardrails['total_cost_guardrail_observed']:
        decision = 'BLOCK'
    elif guardrails['stockout_not_worse'] and (guardrails['wape_improved'] or guardrails['under_forecast_improved']):
        decision = 'ITERATE'
    else:
        decision = 'HOLD'

    return {
        'decision': decision,
        'primary_metric': 'wape',
        'control_forecast': control_forecast,
        'treatment_forecast': treatment_forecast,
        'control_policy': control_policy,
        'treatment_policy': treatment_policy,
        'wape_relative_improvement': wape_rel_improvement,
        'under_forecast_relative_improvement': under_rel_improvement,
        'stockout_delta_treatment_minus_control': stockout_delta,
        'overstock_relative_worsening': overstock_rel_worsening,
        'total_cost_relative_worsening': total_cost_rel_worsening,
        'worst_slice_wape_relative_worsening': worst_slice_rel_worsening,
        'ab_stockout_ci_stable_improvement': ci_stable,
        'guardrails': {k: 'pass' if v else 'fail' for k, v in guardrails.items()},
        'notes': [
            'Offline public-data decision gate. Not a production Amazon release decision.',
            'Inventory and lead-time impacts use synthetic/proxy overlays.',
            'Forecast accuracy guardrails and policy/business guardrails are evaluated separately.',
        ] + ([] if evidence_complete else [
            'Release HOLD: required forecast or inventory policy metrics are missing or non-finite.'
        ]),
    }
