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
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install optional GPU dependencies with: "
        "python -m pip install -r requirements-gpu.txt"
    ) from exc


class TemporalMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def wape(y_true, y_pred) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / max(denom, 1e-9))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    input_path = Path("outputs/demand_mart.csv")
    if not input_path.exists():
        raise FileNotFoundError(
            "outputs/demand_mart.csv not found. Run the main pipeline first: "
            "python -m src.run_pipeline --config configs/pipeline.yaml"
        )

    out_dir = Path("outputs")
    rep_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)

    df = pd.read_csv(input_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    if "demand" not in df.columns:
        raise ValueError("Expected target column `demand` in outputs/demand_mart.csv")

    # Select numeric features only. Exclude known target/prediction/leakage-like columns.
    exclude = {
        "demand",
        "y_true",
        "pred",
        "prediction",
        "forecast",
    }

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and not c.lower().startswith("pred_")
    ]

    # Keep the experiment robust even if the mart schema changes.
    numeric_cols = [
        c for c in numeric_cols
        if df[c].notna().mean() > 0.80 and df[c].nunique(dropna=True) > 1
    ]

    if not numeric_cols:
        raise ValueError("No usable numeric feature columns found for Temporal MLP.")

    data = df[numeric_cols + ["demand"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 500:
        raise ValueError(f"Not enough rows after cleaning: {len(data)}")

    X = data[numeric_cols].astype("float32").to_numpy()
    y = data["demand"].astype("float32").to_numpy()

    # Temporal split by row order after date sorting.
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    model = TemporalMLP(n_features=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    epochs = 15

    start_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            pv = model(xv).detach().cpu().numpy()
        val_wape = wape(y_val, pv)

        if val_wape < best_val:
            best_val = val_wape
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch={epoch:02d} val_wape={val_wape:.4f}")

    train_seconds = time.time() - start_train

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        start_infer = time.time()
        pt = model(xt).detach().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        infer_seconds = time.time() - start_infer

    train_mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))
    naive_metrics = {
        "model": "train_mean_naive",
        "device": "cpu",
        "n_test": int(len(y_test)),
        "wape": wape(y_test, train_mean_pred),
        "rmse": rmse(y_test, train_mean_pred),
        "mean_bias": float(np.mean(train_mean_pred - y_test)),
        "under_forecast_rate": float(np.mean(train_mean_pred < y_test)),
        "train_seconds": 0.0,
        "inference_seconds": 0.0,
    }

    mlp_metrics = {
        "model": "torch_temporal_mlp",
        "device": device,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "epochs": int(epochs),
        "wape": wape(y_test, pt),
        "rmse": rmse(y_test, pt),
        "mean_bias": float(np.mean(pt - y_test)),
        "under_forecast_rate": float(np.mean(pt < y_test)),
        "train_seconds": round(train_seconds, 4),
        "inference_seconds": round(infer_seconds, 6),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
    }

    metrics = pd.DataFrame([naive_metrics, mlp_metrics])
    metrics.to_csv(out_dir / "gpu_temporal_mlp_metrics.csv", index=False)

    with open(out_dir / "gpu_temporal_mlp_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": numeric_cols,
                "target": "demand",
                "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "device": device,
            },
            f,
            indent=2,
        )

    report = [
        "# GPU Temporal MLP Forecasting Smoke Report",
        "",
        "## Purpose",
        "",
        "This lightweight experiment validates that a PyTorch model can train and run inference on the retail forecasting feature mart. It is an optional GPU-backed extension and does not replace the main CPU-runnable pipeline.",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The experiment verifies the GPU model-training path and produces comparable forecasting metrics against a simple train-mean baseline. This is a smoke experiment, not yet the final specialty method.",
        "",
        "## Next step",
        "",
        "Use the GPU model as a candidate forecast or residual-correction model, then evaluate it through the same bias, inventory-impact, A/B-style, and release-gate framework.",
    ]
    (rep_dir / "gpu_temporal_mlp_report.md").write_text("\n".join(report), encoding="utf-8")

    print(metrics.to_string(index=False))
    print("Wrote outputs/gpu_temporal_mlp_metrics.csv")
    print("Wrote reports/gpu_temporal_mlp_report.md")


if __name__ == "__main__":
    main()
