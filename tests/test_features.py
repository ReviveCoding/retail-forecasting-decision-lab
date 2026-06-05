
import pandas as pd
from src.features.demand_state_timestep import add_demand_state_timestep


def test_demand_state_timestep_columns():
    df = pd.DataFrame({
        'item_store_id':['a']*5,
        'date':pd.date_range('2020-01-01', periods=5),
        'demand':[0,1,2,0,3],
        'event_flag':[0,0,1,0,0],
        'cat_id':['c']*5,
    })
    out = add_demand_state_timestep(df)
    assert 'normalized_lifecycle_age' in out.columns
    assert 'intermittency_score' in out.columns
