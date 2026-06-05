from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from src.models.tree_forecaster import train_forecaster, predict_forecaster


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dims=(128, 64), dropout: float = 0.10) -> None:
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _wape(y_true, y_pred) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / max(denom, 1e-9))


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _metric_row(name, y_true, y_pred, device="cpu", train_seconds=0.0, inference_seconds=0.0):
    return {
        "model": name,
        "device": device,
        "n_test": int(len(y_true)),
        "wape": _wape(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mean_bias": float(np.mean(y_pred - y_true)),
        "under_forecast_rate": float(np.mean(y_pred < y_true)),
        "train_seconds": round(float(train_seconds), 4),
        "inference_seconds": round(float(inference_seconds), 6),
    }


def _choose_numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        "demand",
        "pred_tree",
        "pred_corrected",
        "pred_gpu_base_tree",
        "pred_guarded_gpu_residual",
        "pred_gpu_residual_raw",
        "order_signal_risk_safe_corrected",
        "order_signal_guarded_gpu_residual",
        "pred_p10",
        "pred_p90",
    }
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and not c.lower().startswith("pred_")
    ]
    cols = [
        c for c in cols
        if df[c].notna().mean() > 0.80 and df[c].nunique(dropna=True) > 1
    ]
    return cols


def _to_builtin_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def apply_guarded_gpu_residual_stage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    model_cfg: dict,
    gpu_cfg: dict,
    seed: int,
    out_dir: Path,
    rep_dir: Path,
) -> dict:
    if not bool(gpu_cfg.get("enabled", False)):
        return {"test": test, "enabled": False}

    if torch is None:
        raise ImportError("PyTorch is required for gpu_residual.enabled=true. Install requirements-gpu.txt first.")

    out_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)

    split_cfg = gpu_cfg.get("split", {})
    train_base_end = float(split_cfg.get("train_base_end", 0.60))
    residual_fit_end = float(split_cfg.get("residual_fit_end", 0.75))
    train_base_fraction = min(max(train_base_end / max(residual_fit_end, 1e-9), 0.50), 0.95)

    train_sorted = train.sort_values(["date", "item_store_id"]).reset_index(drop=True)
    cut = int(len(train_sorted) * train_base_fraction)
    cut = max(100, min(cut, len(train_sorted) - 100))
    train_base = train_sorted.iloc[:cut].copy()
    residual_fit = train_sorted.iloc[cut:].copy()
    guard_val = val.sort_values(["date", "item_store_id"]).reset_index(drop=True).copy()
    test_out = test.sort_values(["date", "item_store_id"]).reset_index(drop=True).copy()

    base_model = train_forecaster(
        train_base,
        seed=seed,
        n_estimators=model_cfg.get("n_estimators", 80),
        max_depth=model_cfg.get("max_depth", 12),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
    )

    residual_fit["pred_gpu_base_tree"] = predict_forecaster(base_model, residual_fit)
    guard_val["pred_gpu_base_tree"] = predict_forecaster(base_model, guard_val)
    test_out["pred_gpu_base_tree"] = predict_forecaster(base_model, test_out)

    feature_cols = _choose_numeric_features(residual_fit)
    feature_cols = list(dict.fromkeys(feature_cols + ["pred_gpu_base_tree"]))

    required = list(dict.fromkeys(feature_cols + ["demand"]))
    residual_fit_data = residual_fit[required].replace([np.inf, -np.inf], np.nan).dropna()
    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()
    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()
    for name, data_frame in [("residual_fit_data", residual_fit_data), ("guard_data", guard_data), ("test_data", test_data)]:
        if data_frame.columns.duplicated().any():
            dupes = data_frame.columns[data_frame.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns in {name}: {dupes}")
    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()
    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()

    if len(residual_fit_data) < 100 or len(guard_data) < 50 or len(test_data) < 50:
        raise ValueError("Not enough rows for leakage-safe GPU residual stage.")

    X_fit = residual_fit_data[feature_cols].astype("float32").to_numpy()
    y_fit = residual_fit_data["demand"].astype("float32").to_numpy()
    base_fit = residual_fit_data["pred_gpu_base_tree"].astype("float32").to_numpy()
    residual_y = y_fit - base_fit

    X_guard = guard_data[feature_cols].astype("float32").to_numpy()
    y_guard = guard_data["demand"].astype("float32").to_numpy()
    base_guard = guard_data["pred_gpu_base_tree"].astype("float32").to_numpy()

    X_test = test_data[feature_cols].astype("float32").to_numpy()
    y_test = test_data["demand"].astype("float32").to_numpy()
    base_test = test_data["pred_gpu_base_tree"].astype("float32").to_numpy()

    mean = X_fit.mean(axis=0, keepdims=True)
    std = X_fit.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_fit = (X_fit - mean) / std
    X_guard = (X_guard - mean) / std
    X_test = (X_test - mean) / std

    requested_device = str(gpu_cfg.get("device", "cuda"))
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(residual_y, dtype=torch.float32),
        ),
        batch_size=int(gpu_cfg.get("batch_size", 1024)),
        shuffle=True,
    )

    model = ResidualMLP(
        n_features=X_fit.shape[1],
        hidden_dims=tuple(gpu_cfg.get("hidden_dims", [128, 64])),
        dropout=float(gpu_cfg.get("dropout", 0.10)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(gpu_cfg.get("learning_rate", 0.001)),
        weight_decay=float(gpu_cfg.get("weight_decay", 0.0001)),
    )
    loss_fn = nn.SmoothL1Loss()

    best_state = None
    best_guard_wape = float("inf")
    epochs = int(gpu_cfg.get("epochs", 20))
    start_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, rb in train_loader:
            xb = xb.to(device)
            rb = rb.to(device)
            opt.zero_grad(set_to_none=True)
            pred_r = model(xb)
            loss = loss_fn(pred_r, rb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xg = torch.tensor(X_guard, dtype=torch.float32, device=device)
            raw_guard = model(xg).detach().cpu().numpy()
        guard_pred = np.clip(base_guard + raw_guard, 0.0, None)
        guard_wape = _wape(y_guard, guard_pred)
        if guard_wape < best_guard_wape:
            best_guard_wape = guard_wape
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = time.time() - start_train
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xg = torch.tensor(X_guard, dtype=torch.float32, device=device)
        xt = torch.tensor(X_test, dtype=torch.float32, device=device)
        raw_guard = model(xg).detach().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        start_infer = time.time()
        raw_test = model(xt).detach().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.time() - start_infer

    guard_cfg = gpu_cfg.get("guard", {})
    lambda_grid = guard_cfg.get("lambda_grid", [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00])
    clip_grid = guard_cfg.get("clip_quantile_grid", [0.80, 0.90, 0.95])

    grid_rows = []
    base_guard_metrics = _metric_row("gpu_base_tree_guard", y_guard, base_guard, device="cpu")
    raw_guard_pred = np.clip(base_guard + raw_guard, 0.0, None)
    raw_guard_metrics = _metric_row("raw_gpu_residual_guard", y_guard, raw_guard_pred, device=device)

    for lam in lambda_grid:
        for q in clip_grid:
            q = float(q)
            lam = float(lam)
            lo = float(np.quantile(raw_guard, 1.0 - q))
            hi = float(np.quantile(raw_guard, q))
            clipped_guard = np.clip(raw_guard, lo, hi)
            pred_guard = np.clip(base_guard + lam * clipped_guard, 0.0, None)
            row = _metric_row(f"guarded_gpu_lambda_{lam}_q_{q}", y_guard, pred_guard, device=device)
            row.update({"lambda": lam, "clip_quantile": q, "clip_low": lo, "clip_high": hi})
            grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    base_wape = float(base_guard_metrics["wape"])
    raw_bias_abs = abs(float(raw_guard_metrics["mean_bias"]))
    candidates = grid[(grid["wape"] < base_wape) & (grid["mean_bias"].abs() <= max(raw_bias_abs, 1e-9))].copy()
    if candidates.empty:
        candidates = grid.copy()
    candidates["score"] = candidates["wape"].rank() + candidates["mean_bias"].abs().rank() + candidates["under_forecast_rate"].rank()
    selected = candidates.sort_values("score").iloc[0]
    selected_dict = _to_builtin_dict(selected.to_dict())

    clip_low = float(selected_dict["clip_low"])
    clip_high = float(selected_dict["clip_high"])
    lam = float(selected_dict["lambda"])
    pred_raw_test = np.clip(base_test + raw_test, 0.0, None)
    pred_guarded_test = np.clip(base_test + lam * np.clip(raw_test, clip_low, clip_high), 0.0, None)

    test_out["pred_gpu_base_tree"] = predict_forecaster(base_model, test_out)
    test_out["pred_gpu_residual_raw"] = np.nan
    test_out["pred_guarded_gpu_residual"] = np.nan
    valid_idx = test_data.index
    test_out.loc[valid_idx, "pred_gpu_residual_raw"] = pred_raw_test
    test_out.loc[valid_idx, "pred_guarded_gpu_residual"] = pred_guarded_test
    test_out["pred_guarded_gpu_residual"] = test_out["pred_guarded_gpu_residual"].fillna(test_out["pred_gpu_base_tree"])
    test_out["pred_gpu_residual_raw"] = test_out["pred_gpu_residual_raw"].fillna(test_out["pred_gpu_base_tree"])
    # Forecast metric and ordering-policy signal are intentionally separated.
    # The guarded point forecast is used for WAPE evaluation.
    # The release-safe signal floors the GPU forecast at the tree forecast to reduce
    # under-forecast and stockout risk without using the test window for tuning.
    if "pred_tree" in test_out.columns:
        test_out["pred_release_safe_gpu_residual"] = test_out[["pred_guarded_gpu_residual", "pred_tree"]].max(axis=1)
    else:
        test_out["pred_release_safe_gpu_residual"] = test_out[["pred_guarded_gpu_residual", "pred_gpu_base_tree"]].max(axis=1)

    test_out["order_signal_guarded_gpu_residual"] = test_out["pred_guarded_gpu_residual"]
    test_out["order_signal_release_safe_gpu_residual"] = test_out["pred_release_safe_gpu_residual"]

    test_metrics = pd.DataFrame([
        _metric_row("gpu_base_tree", y_test, base_test, device="cpu"),
        _metric_row("raw_gpu_residual_corrected", y_test, pred_raw_test, device=device, train_seconds=train_seconds, inference_seconds=inference_seconds),
        _metric_row("guarded_gpu_residual_corrected", y_test, pred_guarded_test, device=device, train_seconds=train_seconds, inference_seconds=inference_seconds),
    ])

    grid.to_csv(out_dir / "gpu_integrated_guard_grid.csv", index=False)
    test_metrics.to_csv(out_dir / "gpu_integrated_metrics.csv", index=False)
    with open(out_dir / "gpu_integrated_selected_guard.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_guard": selected_dict,
            "device": device,
            "feature_columns": feature_cols,
            "split": {
                "train_base_rows": int(len(train_base)),
                "residual_fit_rows": int(len(residual_fit)),
                "guard_val_rows": int(len(guard_val)),
                "test_rows": int(len(test_out)),
            },
            "claim_boundary": "Leakage-safe stage: residual model fit on residual_fit, guard selected on val, test used only for final evaluation.",
        }, f, indent=2)

    report = [
        "# Leakage-Safe Guarded GPU Residual Corrector",
        "",
        "## Purpose",
        "",
        "This integrated stage trains a CUDA-backed PyTorch residual model without using the test window for fitting, scaling, early stopping target construction, or guard selection.",
        "",
        "## Split design",
        "",
        f"- train_base rows: {len(train_base)}",
        f"- residual_fit rows: {len(residual_fit)}",
        f"- guard_val rows: {len(guard_val)}",
        f"- test rows: {len(test_out)}",
        "",
        "## Test metrics",
        "",
        test_metrics.to_markdown(index=False),
        "",
        "## Selected guard",
        "",
        pd.DataFrame([selected_dict]).to_markdown(index=False),
        "",
        "## Claim boundary",
        "",
        "The residual model is fit on residual_fit only. Lambda and clipping are selected on guard_val only. Test is used only once for final evaluation.",
    ]
    (rep_dir / "gpu_integrated_guarded_residual_report.md").write_text("\n".join(report), encoding="utf-8")

    return {
        "test": test_out,
        "enabled": True,
        "metrics": test_metrics,
        "selected_guard": selected_dict,
        "device": device,
    }
