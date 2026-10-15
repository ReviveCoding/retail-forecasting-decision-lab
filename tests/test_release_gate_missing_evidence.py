"""Release decisions require both forecast and inventory-policy evidence."""

import pandas as pd

from src.evaluation.release_gate import build_release_gate


def _forecast_metrics():
    return pd.DataFrame(
        [
            {"model_name": "tree_ensemble", "wape": 0.20, "under_forecast_rate": 0.40},
            {"model_name": "corrected", "wape": 0.17, "under_forecast_rate": 0.30},
        ]
    )


def _inventory_metrics():
    return pd.DataFrame(
        [
            {
                "forecast_model": "tree_ensemble_policy",
                "lead_time": 7,
                "safety_stock_factor": 0.5,
                "stockout_rate": 0.25,
                "overstock_units": 100,
                "total_cost": 300,
            },
            {
                "forecast_model": "risk_safe_corrected_policy",
                "lead_time": 7,
                "safety_stock_factor": 0.5,
                "stockout_rate": 0.20,
                "overstock_units": 101,
                "total_cost": 297,
            },
        ]
    )


def test_complete_evidence_preserves_launch_decision():
    result = build_release_gate(_forecast_metrics(), _inventory_metrics(), pd.DataFrame())
    assert result["decision"] == "LAUNCH"
    assert not any("required forecast or inventory policy metrics" in n for n in result["notes"])


def test_missing_inventory_evidence_cannot_launch():
    inventory = pd.DataFrame(columns=_inventory_metrics().columns)
    result = build_release_gate(_forecast_metrics(), inventory, pd.DataFrame())
    assert result["decision"] == "HOLD"
    assert any("required forecast or inventory policy metrics" in n for n in result["notes"])


def test_missing_required_inventory_metric_cannot_launch():
    inventory = _inventory_metrics().drop(columns=["total_cost"])
    result = build_release_gate(_forecast_metrics(), inventory, pd.DataFrame())
    assert result["decision"] == "HOLD"


def test_missing_forecast_metrics_cannot_launch():
    result = build_release_gate(pd.DataFrame(), _inventory_metrics(), pd.DataFrame())
    assert result["decision"] == "HOLD"


def test_nonfinite_forecast_metric_cannot_launch():
    metrics = _forecast_metrics()
    metrics.loc[0, "wape"] = float("nan")
    result = build_release_gate(metrics, _inventory_metrics(), pd.DataFrame())
    assert result["decision"] == "HOLD"
