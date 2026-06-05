
import pandas as pd
from src.data.validate_schema import validate_panel_schema


def test_schema_validator():
    df = pd.DataFrame({
        'item_store_id':['i_s'], 'item_id':['i'], 'store_id':['s'], 'state_id':['CA'], 'cat_id':['c'], 'dept_id':['d'],
        'date':pd.to_datetime(['2020-01-01']), 'demand':[1], 'sell_price':[1.0], 'event_flag':[0], 'snap_flag':[0]
    })
    report = validate_panel_schema(df)
    assert report['schema_pass'] is True
