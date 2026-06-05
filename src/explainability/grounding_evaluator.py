from __future__ import annotations
import re
import pandas as pd

_NUMBER_PATTERNS = {
    'tree_forecast': re.compile(r"tree=([-+]?\d*\.?\d+)"),
    'corrected_forecast': re.compile(r"corrected=([-+]?\d*\.?\d+)"),
    'actual': re.compile(r"actual=([-+]?\d*\.?\d+)"),
}
_UNSUPPORTED_TERMS = ['weather', 'vendor outage', 'competitor', 'supply chain strike', 'macroeconomic shock']


def _number_matches(text: str, value: float, pattern: re.Pattern[str], tol: float = 0.05) -> bool:
    m = pattern.search(text)
    if not m:
        return False
    try:
        return abs(float(m.group(1)) - float(value)) <= tol
    except ValueError:
        return False


def evaluate_explanation_grounding(explanations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for _, r in explanations.iterrows():
        text = str(r['explanation'])
        lower = text.lower()
        numeric_consistency = all(
            _number_matches(text, float(r[col]), pattern)
            for col, pattern in _NUMBER_PATTERNS.items()
            if col in r.index
        )
        driver = str(r.get('top_global_feature', ''))
        has_driver = ('Top global driver:' in text and driver in text) or ('residual correction' in lower) or ('demand-state' in lower)
        uncertainty_disclosure = any(term in lower for term in ['residual correction', 'demand-state', 'uncertainty', 'disagreement'])
        unsupported_external_claim = any(term in lower for term in _UNSUPPORTED_TERMS)
        unsupported_claim = unsupported_external_claim or not (numeric_consistency and has_driver)
        rows.append({
            'case_type': r.get('case_type', 'unknown'),
            'item_store_id': r['item_store_id'],
            'date': r['date'],
            'numeric_consistency': bool(numeric_consistency),
            'driver_direction_correctness': bool(has_driver and not unsupported_external_claim),
            'uncertainty_disclosure': bool(uncertainty_disclosure),
            'unsupported_claim_flag': bool(unsupported_claim),
            'groundedness_score': float((numeric_consistency + (has_driver and not unsupported_external_claim) + uncertainty_disclosure) / 3),
        })
    detail = pd.DataFrame(rows)
    metrics = {
        'n_explanations': int(len(detail)),
        'n_negative_controls': int((detail['case_type'] == 'negative_control').sum()) if len(detail) else 0,
        'numeric_consistency_rate': float(detail['numeric_consistency'].mean()) if len(detail) else 0.0,
        'driver_direction_correctness_rate': float(detail['driver_direction_correctness'].mean()) if len(detail) else 0.0,
        'uncertainty_disclosure_rate': float(detail['uncertainty_disclosure'].mean()) if len(detail) else 0.0,
        'unsupported_claim_rate': float(detail['unsupported_claim_flag'].mean()) if len(detail) else 0.0,
        'negative_control_detection_rate': float(detail.loc[detail['case_type'] == 'negative_control', 'unsupported_claim_flag'].mean()) if (detail['case_type'] == 'negative_control').any() else 0.0,
        'mean_groundedness_score': float(detail['groundedness_score'].mean()) if len(detail) else 0.0,
    }
    return detail, metrics
