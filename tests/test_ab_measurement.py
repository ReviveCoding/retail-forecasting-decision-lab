import pandas as pd
from src.experiments.ab_style_measurement import build_ab_style_readout


def test_ab_readout_produces_rows():
    rows=[]
    for model, stockout in [('tree_ensemble_policy',1),('risk_safe_corrected_policy',0)]:
        for i in range(4):
            for d in range(7):
                rows.append({
                    'forecast_model':model,'lead_time':7,'safety_stock_factor':0.5,
                    'item_store_id':f'id_{i}','date':pd.Timestamp('2026-01-01') + pd.Timedelta(days=d),
                    'stockout_flag':stockout,'service_level':1-stockout*0.1,'total_cost':1.0,
                    'demand_coverage_f1':0.9,'inventory_precision':0.8,'overstock_units':0.1
                })
    out = build_ab_style_readout(pd.DataFrame(rows), n_bootstrap=5)
    assert not out.empty
    assert set(out['unit_type']) == {'item-store-week'}
    assert 'n_clusters' in out.columns
