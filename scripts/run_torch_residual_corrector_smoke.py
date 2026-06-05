from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def wape(y_true, y_pred) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / max(denom, 1e-9))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metrics(name, y_true, y_pred, device="cpu", train_seconds=0.0, inference_seconds=0.0):
    return {
        "model": name,
        "device": device,
        "n_test": int(len(y_true)),
        "wape": wape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mean_bias": float(np.mean(y_pred - y_true)),
        "under_forecast_rate": float(np.mean(y_pred < y_true)),
        "train_seconds": round(float(train_seconds), 4),
        "inference_seconds": round(float(inference_seconds), 6),
    }


def main() -> None:
    in_path = Path("outputs/corrected_forecasts_sample.csv")
    if not in_path.exists():
        raise FileNotFoundError("Run the main pipeline first so outputs/corrected_forecasts_sample.csv exists.")

    out_dir = Path("outputs")
    rep_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)

    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "item_store_id"]).reset_index(drop=True)

    required = {"demand", "pred_tree", "pred_corrected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    exclude = {
        "demand",
        "pred_tree",
        "pred_corrected",
        "order_signal_risk_safe_corrected",
        "pred_p10",
        "pred_p90",
    }

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    numeric_cols = [
        c for c in numeric_cols
        if df[c].notna().mean() > 0.80 and df[c].nunique(dropna=True) > 1
    ]

    data = df[numeric_cols + ["demand", "pred_tree", "pred_corrected"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 500:
        raise ValueError(f"Not enough rows after cleaning: {len(data)}")

    X = data[numeric_cols].astype("float32").to_numpy()
    y = data["demand"].astype("float32").to_numpy()
    pred_tree = data["pred_tree"].astype("float32").to_numpy()
    pred_corrected = data["pred_corrected"].astype("float32").to_numpy()
    residual = y - pred_tree

    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, r_train = X[:train_end], residual[:train_end]
    X_val, r_val = X[train_end:val_end], residual[train_end:val_end]
    X_test = X[val_end:]

    y_test = y[val_end:]
    pred_tree_test = pred_tree[val_end:]
    pred_corrected_test = pred_corrected[val_end:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(r_train, dtype=torch.float32),
        ),
        batch_size=1024,
        shuffle=True,
    )

    model = ResidualMLP(n_features=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    epochs = 20

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
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            pred_r_val = model(xv).detach().cpu().numpy()

        val_pred = pred_tree[train_end:val_end] + pred_r_val
        val_pred = np.clip(val_pred, 0.0, None)
        val_wape = wape(y[train_end:val_end], val_pred)

        if val_wape < best_val:
            best_val = val_wape
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch={epoch:02d} val_corrected_wape={val_wape:.4f}")

    train_seconds = time.time() - start_train

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start_infer = time.time()
        pred_residual = model(xt).detach().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()

        inference_seconds = time.time() - start_infer

    pred_torch_residual = np.clip(pred_tree_test + pred_residual, 0.0, None)

    rows = [
        metrics("tree_ensemble", y_test, pred_tree_test),
        metrics("current_residual_corrected", y_test, pred_corrected_test),
        metrics(
            "torch_gpu_residual_corrected",
            y_test,
            pred_torch_residual,
            device=device,
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
        ),
    ]

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "gpu_residual_corrector_metrics.csv", index=False)

    pred_out = pd.DataFrame({
        "y_true": y_test,
        "pred_tree": pred_tree_test,
        "pred_current_corrected": pred_corrected_test,
        "pred_torch_gpu_residual_corrected": pred_torch_residual,
        "torch_predicted_residual": pred_residual,
    })
    pred_out.to_csv(out_dir / "gpu_residual_corrector_predictions.csv", index=False)

    with open(out_dir / "gpu_residual_corrector_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": numeric_cols,
                "target": "demand_minus_pred_tree",
                "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "device": device,
                "claim_boundary": "Exploratory GPU residual-correction smoke experiment on generated forecast artifact.",
            },
            f,
            indent=2,
        )

    report = [
        "# GPU Residual Corrector Smoke Report",
        "",
        "## Purpose",
        "",
        "This experiment evaluates whether a CUDA-backed PyTorch MLP can learn residual structure on the enriched forecast artifact and improve a tree-ensemble forecast.",
        "",
        "## Metrics",
        "",
        result.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This is an exploratory smoke experiment. It validates the GPU residual-correction training path using feature-rich generated artifacts. Final resume-safe claims should come from a leakage-safe integration inside the main train/validation/test pipeline.",
        "",
        "## Claim boundary",
        "",
        "Do not describe this as a production or final launch result. Use it as GPU ML feasibility evidence until integrated into the core pipeline.",
    ]
    (rep_dir / "gpu_residual_corrector_report.md").write_text("\n".join(report), encoding="utf-8")

    print(result.to_string(index=False))
    print("Wrote outputs/gpu_residual_corrector_metrics.csv")
    print("Wrote reports/gpu_residual_corrector_report.md")


if __name__ == "__main__":
    main()
