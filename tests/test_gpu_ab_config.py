"""Regression tests for GPU policy comparisons in the offline A/B readout."""

import pandas as pd

from src.experiments import ab_style_measurement as measurement


def test_gpu_ab_uses_configured_experiment_settings(monkeypatch):
    captured = {}

    def fake_readout(detail, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"metric_name": "stockout_flag"}])

    monkeypatch.setattr(measurement, "build_ab_style_readout", fake_readout)
    output = measurement.build_gpu_ab_style_readout(
        pd.DataFrame(),
        {
            "tree_ensemble_policy": "pred_tree",
            "guarded_gpu_residual_policy": "guarded",
            "release_safe_gpu_residual_policy": "release_safe",
        },
        {
            "lead_time": 14,
            "safety_stock_factor": 1.0,
            "n_bootstrap": 45,
            "random_seed": 123,
            "unit_type": "item-store-week",
        },
        seed=42,
    )

    assert output is not None and not output.empty
    assert captured == {
        "control": "tree_ensemble_policy",
        "treatment": "release_safe_gpu_residual_policy",
        "lead_time": 14,
        "safety_stock_factor": 1.0,
        "n_bootstrap": 45,
        "seed": 123,
        "unit_type": "item-store-week",
    }


def test_gpu_ab_returns_none_without_a_gpu_policy(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Base A/B evaluator should not run")

    monkeypatch.setattr(measurement, "build_ab_style_readout", unexpected)
    assert measurement.build_gpu_ab_style_readout(
        pd.DataFrame(), {"tree_ensemble_policy": "pred_tree"}, {}, seed=42
    ) is None


def test_gpu_ab_keeps_priority_and_default_settings(monkeypatch):
    captured = {}

    def fake_readout(detail, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(measurement, "build_ab_style_readout", fake_readout)
    measurement.build_gpu_ab_style_readout(
        pd.DataFrame(),
        {
            "guarded_gpu_residual_policy": "guarded",
            "learned_abc_gpu_residual_policy": "learned",
            "learned_abc_release_safe_policy": "safe",
        },
        {},
        seed=29,
    )
    assert captured["treatment"] == "learned_abc_release_safe_policy"
    assert captured["lead_time"] == 7
    assert captured["safety_stock_factor"] == 0.5
    assert captured["n_bootstrap"] == 200
    assert captured["seed"] == 29
