import pandas as pd
from src.explainability.grounding_evaluator import evaluate_explanation_grounding


def test_grounding_detects_negative_control():
    df = pd.DataFrame([
        {
            'case_type':'supported','item_store_id':'id_1','date':'2026-01-01',
            'actual': 10.0, 'tree_forecast': 9.0, 'corrected_forecast': 11.0,
            'top_global_feature':'lag_28',
            'explanation':'Corrected forecast uses residual correction and demand-state features. Top global driver: lag_28. tree=9.00, corrected=11.00, actual=10.00.'
        },
        {
            'case_type':'negative_control','item_store_id':'id_2','date':'2026-01-02',
            'actual': 10.0, 'tree_forecast': 9.0, 'corrected_forecast': 11.0,
            'top_global_feature':'lag_28',
            'explanation':'Forecast changed because weather and vendor outage happened. tree=100.00, corrected=11.00, actual=10.00.'
        }
    ])
    detail, metrics = evaluate_explanation_grounding(df)
    assert metrics['negative_control_detection_rate'] == 1.0
    assert detail.loc[detail['case_type']=='negative_control','unsupported_claim_flag'].iloc[0]
