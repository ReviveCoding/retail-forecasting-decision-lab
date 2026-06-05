import pandas as pd
from src.models.residual_bias_corrector import train_residual_corrector, apply_residual_correction


def test_residual_correction_outputs_nonnegative():
    df = pd.DataFrame({
        'demand':[1,2,3,4,5,6,7,8,9,10],
        'pred_tree':[1,1,2,3,5,5,7,7,8,9],
        'lag_7':[1]*10,'lag_14':[1]*10,'lag_28':[1]*10,
        'rolling_mean_7':[1]*10,'rolling_std_7':[0]*10,'rolling_mean_28':[1]*10,'rolling_std_28':[0]*10,
    })
    c = train_residual_corrector(df, 'pred_tree')
    p = apply_residual_correction(c, df, 'pred_tree')
    assert len(p) == len(df)
    assert (p >= 0).all()
