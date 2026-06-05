
from __future__ import annotations
import numpy as np
import pandas as pd
from .demand_coverage_metrics import add_demand_coverage_metrics


def simulate_inventory_policies(test_df: pd.DataFrame, pred_cols: dict[str,str], lead_times=(1,3,7), safety_factors=(0.0,0.5,1.0), holding_cost=0.25, lost_sales_cost=1.0, labor_units_per_demand_unit=0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for policy_name, pred_col in pred_cols.items():
        if pred_col not in test_df.columns:
            continue
        for lead_time in lead_times:
            for sf in safety_factors:
                tmp = test_df[['item_store_id','item_id','store_id','cat_id','state_id','date','demand',pred_col,'rolling_std_28']].copy()
                tmp = tmp.rename(columns={'demand':'actual_demand', pred_col:'forecast_demand'})
                safety_stock = sf * tmp['rolling_std_28'].fillna(0) * np.sqrt(max(float(lead_time), 1.0))
                tmp['forecast_model'] = policy_name
                tmp['lead_time'] = lead_time
                tmp['safety_stock_factor'] = sf
                tmp['safety_stock'] = safety_stock.clip(lower=0)
                tmp['order_qty'] = (tmp['forecast_demand'] + tmp['safety_stock']).clip(lower=0)
                tmp['stockout_flag'] = (tmp['order_qty'] < tmp['actual_demand']).astype(int)
                tmp['overstock_units'] = (tmp['order_qty'] - tmp['actual_demand']).clip(lower=0)
                tmp['service_level'] = np.minimum(tmp['order_qty'], tmp['actual_demand']) / tmp['actual_demand'].replace(0, np.nan)
                tmp['service_level'] = tmp['service_level'].fillna(1.0).clip(upper=1.0)
                tmp['holding_cost'] = tmp['overstock_units'] * holding_cost
                tmp['lost_sales_proxy'] = (tmp['actual_demand'] - tmp['order_qty']).clip(lower=0) * lost_sales_cost
                tmp['total_cost'] = tmp['holding_cost'] + tmp['lost_sales_proxy']
                tmp['labor_planning_error_proxy'] = (tmp['forecast_demand'] - tmp['actual_demand']).abs() * labor_units_per_demand_unit
                tmp = add_demand_coverage_metrics(tmp)
                detail_rows.append(tmp)
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(['forecast_model','lead_time','safety_stock_factor'], as_index=False).agg(
        n_obs=('actual_demand','size'),
        stockout_rate=('stockout_flag','mean'),
        service_level=('service_level','mean'),
        overstock_units=('overstock_units','mean'),
        holding_cost=('holding_cost','mean'),
        lost_sales_proxy=('lost_sales_proxy','mean'),
        total_cost=('total_cost','mean'),
        labor_planning_error_proxy=('labor_planning_error_proxy','mean'),
        demand_recall=('demand_recall','mean'),
        inventory_precision=('inventory_precision','mean'),
        demand_coverage_f1=('demand_coverage_f1','mean'),
    )
    return detail, summary
