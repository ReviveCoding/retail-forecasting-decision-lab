import pandas as pd
from src.evaluation.release_gate import build_release_gate


def test_release_gate_returns_decision():
    metrics = pd.DataFrame([
        {'model_name':'tree_ensemble','wape':0.2,'under_forecast_rate':0.5},
        {'model_name':'corrected','wape':0.18,'under_forecast_rate':0.4},
    ])
    inv = pd.DataFrame([
        {'forecast_model':'tree_ensemble_policy','lead_time':7,'safety_stock_factor':0.5,'stockout_rate':0.2,'overstock_units':1.0,'total_cost':1.0},
        {'forecast_model':'risk_safe_corrected_policy','lead_time':7,'safety_stock_factor':0.5,'stockout_rate':0.18,'overstock_units':1.01,'total_cost':1.0},
    ])
    gate = build_release_gate(metrics, inv, pd.DataFrame(), thresholds={'max_overstock_relative_worsening':0.02})
    assert gate['decision'] in {'LAUNCH','ITERATE','HOLD','BLOCK'}


def test_release_gate_overstock_guardrail_can_fail():
    metrics = pd.DataFrame([
        {'model_name':'tree_ensemble','wape':0.2,'under_forecast_rate':0.5},
        {'model_name':'corrected','wape':0.18,'under_forecast_rate':0.4},
    ])
    inv = pd.DataFrame([
        {'forecast_model':'tree_ensemble_policy','lead_time':7,'safety_stock_factor':0.5,'stockout_rate':0.2,'overstock_units':1.0,'total_cost':1.0},
        {'forecast_model':'risk_safe_corrected_policy','lead_time':7,'safety_stock_factor':0.5,'stockout_rate':0.18,'overstock_units':2.0,'total_cost':1.0},
    ])
    gate = build_release_gate(metrics, inv, pd.DataFrame(), thresholds={'max_overstock_relative_worsening':0.02})
    assert gate['guardrails']['overstock_guardrail_observed'] == 'fail'
    assert gate['decision'] == 'BLOCK'
