
from __future__ import annotations
import pandas as pd

REQUIRED_PANEL_COLUMNS = {
    'item_store_id','item_id','store_id','state_id','cat_id','dept_id',
    'date','demand','sell_price','event_flag','snap_flag'
}


def validate_panel_schema(panel: pd.DataFrame) -> dict:
    missing = sorted(REQUIRED_PANEL_COLUMNS.difference(panel.columns))
    duplicate_keys = int(panel.duplicated(['item_store_id','date']).sum()) if not missing else None
    null_counts = {c: int(panel[c].isna().sum()) for c in panel.columns if c in REQUIRED_PANEL_COLUMNS}
    date_gaps = 0
    if 'item_store_id' in panel.columns and 'date' in panel.columns:
        for _, g in panel.groupby('item_store_id'):
            diffs = g.sort_values('date')['date'].diff().dropna().dt.days
            date_gaps += int((diffs > 1).sum())
    return {
        'n_rows': int(len(panel)),
        'n_series': int(panel['item_store_id'].nunique()) if 'item_store_id' in panel.columns else 0,
        'missing_required_columns': missing,
        'duplicate_item_store_date_keys': duplicate_keys,
        'date_gap_count': int(date_gaps),
        'null_counts_required': null_counts,
        'schema_pass': (not missing) and duplicate_keys == 0,
    }
