from pathlib import Path
import pandas as pd
import numpy as np


def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / max(denom, 1e-9))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def row(name, y, pred):
    return {
        "model": name,
        "wape": wape(y, pred),
        "rmse": rmse(y, pred),
        "mean_bias": float(np.mean(pred - y)),
        "under_forecast_rate": float(np.mean(pred < y)),
    }


in_path = Path("outputs/gpu_residual_corrector_predictions.csv")
if not in_path.exists():
    raise FileNotFoundError("Run scripts/run_torch_residual_corrector_smoke.py first.")

df = pd.read_csv(in_path)

y = df["y_true"].to_numpy(dtype=float)
pred_tree = df["pred_tree"].to_numpy(dtype=float)
pred_current = df["pred_current_corrected"].to_numpy(dtype=float)
raw_residual = df["torch_predicted_residual"].to_numpy(dtype=float)

rows = [
    row("tree_ensemble", y, pred_tree),
    row("current_residual_corrected", y, pred_current),
    row("torch_gpu_residual_corrected_raw", y, df["pred_torch_gpu_residual_corrected"].to_numpy(dtype=float)),
]

# Guarded post-processing grid.
# lambda controls correction strength.
# q controls residual clipping based on residual prediction quantiles.
for lam in [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]:
    for q in [0.80, 0.90, 0.95]:
        lo = np.quantile(raw_residual, 1 - q)
        hi = np.quantile(raw_residual, q)
        clipped = np.clip(raw_residual, lo, hi)
        pred = np.clip(pred_tree + lam * clipped, 0.0, None)

        r = row(f"guarded_torch_residual_lambda_{lam}_q_{q}", y, pred)
        r["lambda"] = lam
        r["clip_quantile"] = q
        rows.append(r)

result = pd.DataFrame(rows)

# Choose a balanced candidate:
# 1. WAPE better than tree
# 2. under-forecast rate no worse than raw torch correction
# 3. mean bias closer to zero than raw torch correction
tree = result[result["model"].eq("tree_ensemble")].iloc[0]
raw = result[result["model"].eq("torch_gpu_residual_corrected_raw")].iloc[0]

candidates = result[
    result["model"].str.startswith("guarded_torch")
    & (result["wape"] < tree["wape"])
    & (result["under_forecast_rate"] <= raw["under_forecast_rate"])
    & (result["mean_bias"].abs() <= abs(raw["mean_bias"]))
].copy()

if len(candidates) > 0:
    candidates["score"] = (
        candidates["wape"].rank()
        + candidates["under_forecast_rate"].rank()
        + candidates["mean_bias"].abs().rank()
    )
    selected = candidates.sort_values("score").iloc[0]
else:
    result["score"] = (
        result["wape"].rank()
        + result["under_forecast_rate"].rank()
        + result["mean_bias"].abs().rank()
    )
    selected = result[result["model"].str.startswith("guarded_torch")].sort_values("score").iloc[0]

out_dir = Path("outputs")
rep_dir = Path("reports")
out_dir.mkdir(exist_ok=True)
rep_dir.mkdir(exist_ok=True)

result.to_csv(out_dir / "guarded_gpu_residual_policy_grid.csv", index=False)
pd.DataFrame([selected]).to_csv(out_dir / "selected_guarded_gpu_residual_policy.csv", index=False)

report = [
    "# Guarded GPU Residual Correction Report",
    "",
    "## Purpose",
    "",
    "This post-processing experiment evaluates whether GPU-predicted residuals can be clipped and scaled to preserve WAPE improvement while reducing business-risk side effects such as excessive under-forecasting or bias shift.",
    "",
    "## Baselines and raw GPU correction",
    "",
    result[result["model"].isin([
        "tree_ensemble",
        "current_residual_corrected",
        "torch_gpu_residual_corrected_raw",
    ])].to_markdown(index=False),
    "",
    "## Selected guarded candidate",
    "",
    pd.DataFrame([selected]).to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    "Raw GPU residual correction improved WAPE but increased under-forecast risk. The guarded variant searches correction strength and residual clipping levels to find a more balanced correction policy.",
    "",
    "## Claim boundary",
    "",
    "This is still a post-hoc smoke experiment on generated artifacts. Resume-safe final claims should come from integrating guarded correction into the main train/validation/test pipeline.",
]

(rep_dir / "guarded_gpu_residual_correction_report.md").write_text("\n".join(report), encoding="utf-8")

print("Wrote outputs/guarded_gpu_residual_policy_grid.csv")
print("Wrote outputs/selected_guarded_gpu_residual_policy.csv")
print("Wrote reports/guarded_gpu_residual_correction_report.md")
print()
print("Selected guarded candidate:")
print(pd.DataFrame([selected]).to_string(index=False))
