
import pandas as pd
from src.evaluation.forecast_metrics import forecast_metric_row


def test_forecast_metric_row_basic():
    df = pd.DataFrame({'demand':[1,2,3], 'pred':[1,3,2]})
    row = forecast_metric_row(df, 'pred', 'm')
    assert row['n_obs'] == 3
    assert row['wape'] > 0
    assert 0 <= row['under_forecast_rate'] <= 1
