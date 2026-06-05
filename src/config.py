from __future__ import annotations
from pathlib import Path
from typing import Any
from src.utils import load_yaml


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _legacy_to_nested(cfg: dict[str, Any]) -> dict[str, Any]:
    """Support the older flat configs/data.yaml format for local users."""
    if any(k in cfg for k in ["data", "model", "inventory", "experiment", "release_gate"]):
        return cfg
    return {
        "data": {
            "m5_zip_path": cfg.get("m5_zip_path", "/mnt/data/m5-forecasting-accuracy.zip"),
            "sample_mode": cfg.get("sample_mode", True),
            "n_items": cfg.get("n_items", 30),
            "n_stores": cfg.get("n_stores", 2),
            "max_days": cfg.get("max_days", 420),
            "forecast_horizon_days": cfg.get("forecast_horizon_days", 28),
            "validation_horizon_days": cfg.get("validation_horizon_days", 28),
            "random_seed": cfg.get("random_seed", 42),
            "output_dir": cfg.get("output_dir", "outputs"),
            "report_dir": cfg.get("report_dir", "reports"),
        },
        "model": {"random_seed": cfg.get("random_seed", 42)},
        "inventory": {},
        "experiment": {"random_seed": cfg.get("random_seed", 42)},
        "release_gate": {},
    }


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    cfg = _legacy_to_nested(load_yaml(path))
    defaults = load_yaml(Path("configs/pipeline.yaml")) if Path("configs/pipeline.yaml").exists() else {}
    defaults = _legacy_to_nested(defaults) if defaults else {}
    return _merge_dict(defaults, cfg) if defaults else cfg
