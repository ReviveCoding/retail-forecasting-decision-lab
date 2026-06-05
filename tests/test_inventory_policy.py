
import pandas as pd
from src.simulation.demand_coverage_metrics import add_demand_coverage_metrics


def test_demand_coverage_metrics():
    df = pd.DataFrame({'order_qty':[8,12], 'actual_demand':[10,10]})
    out = add_demand_coverage_metrics(df)
    assert round(out.loc[0,'demand_recall'], 2) == 0.80
    assert round(out.loc[1,'inventory_precision'], 2) == 0.83
    assert 'demand_coverage_f1' in out.columns
