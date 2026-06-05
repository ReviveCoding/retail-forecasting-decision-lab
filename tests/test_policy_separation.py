import pandas as pd
from src.evaluation.forecast_metrics import evaluate_forecasts


def test_risk_safe_policy_not_required_for_forecast_metrics():
    df = pd.DataFrame({'demand':[1,2,3], 'pred_tree':[1,2,2], 'pred_corrected':[1,2,3], 'order_signal_risk_safe_corrected':[2,2,4]})
    metrics = evaluate_forecasts(df, {'tree_ensemble':'pred_tree','corrected':'pred_corrected'})
    assert 'risk_safe_corrected_policy' not in set(metrics['model_name'])
