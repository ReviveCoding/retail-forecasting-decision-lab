from __future__ import annotations
import pandas as pd


def generate_explanations(
    test_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    n_cases: int = 25,
    include_negative_controls: bool = True,
) -> pd.DataFrame:
    top_feature = feature_importance.iloc[0]['feature'] if feature_importance is not None and not feature_importance.empty else 'historical demand features'
    df = test_df.copy()
    df['abs_error_corrected'] = (df['pred_corrected'] - df['demand']).abs()
    cases = df.sort_values('abs_error_corrected', ascending=False).head(n_cases).copy()
    rows = []
    for _, r in cases.iterrows():
        direction = 'higher' if r['pred_corrected'] >= r['pred_tree'] else 'lower'
        explanation = (
            f"Corrected forecast is {direction} than the tree-ensemble forecast for {r['item_store_id']} on {r['date'].date()} "
            f"because residual correction and demand-state features were applied. Top global driver: {top_feature}. "
            f"tree={r['pred_tree']:.2f}, corrected={r['pred_corrected']:.2f}, actual={r['demand']:.2f}."
        )
        rows.append({
            'case_type': 'supported',
            'item_store_id': r['item_store_id'],
            'date': r['date'],
            'actual': float(r['demand']),
            'tree_forecast': float(r['pred_tree']),
            'corrected_forecast': float(r['pred_corrected']),
            'top_global_feature': top_feature,
            'explanation': explanation,
        })

    if include_negative_controls and len(cases) > 0:
        # Add deliberately flawed explanations so the grounding evaluator proves
        # that it can reject unsupported or numerically inconsistent narratives.
        for i, (_, r) in enumerate(cases.head(5).iterrows()):
            if i % 2 == 0:
                bad_text = (
                    f"Forecast moved because weather and a vendor outage changed demand. "
                    f"tree={r['pred_tree'] + 99:.2f}, corrected={r['pred_corrected']:.2f}, actual={r['demand']:.2f}."
                )
            else:
                bad_text = (
                    f"Corrected forecast changed for {r['item_store_id']} without numeric support or a model driver."
                )
            rows.append({
                'case_type': 'negative_control',
                'item_store_id': r['item_store_id'],
                'date': r['date'],
                'actual': float(r['demand']),
                'tree_forecast': float(r['pred_tree']),
                'corrected_forecast': float(r['pred_corrected']),
                'top_global_feature': top_feature,
                'explanation': bad_text,
            })
    return pd.DataFrame(rows)
