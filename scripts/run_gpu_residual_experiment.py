from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REP = ROOT / "reports"


def run_step(name: str, script: str) -> None:
    script_path = ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found for {name}: {script_path}")

    print(f"\n=== {name} ===")
    cmd = [sys.executable, str(script_path)]
    completed = subprocess.run(cmd, cwd=ROOT, text=True)

    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with return code {completed.returncode}")


def pct_improvement(base: float, new: float) -> float:
    return float((base - new) / max(abs(base), 1e-12) * 100.0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    required_inputs = [
        OUT / "corrected_forecasts_sample.csv",
    ]
    for p in required_inputs:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required input: {p}. Run main pipeline first:\n"
                "python -m src.run_pipeline --config configs/pipeline.yaml"
            )

    run_step("PyTorch CUDA validation", "scripts/validate_pytorch_gpu.py")
    run_step("GPU residual corrector smoke experiment", "scripts/run_torch_residual_corrector_smoke.py")
    run_step("Guarded GPU residual postprocess", "scripts/run_guarded_gpu_residual_postprocess.py")

    residual_metrics_path = OUT / "gpu_residual_corrector_metrics.csv"
    selected_path = OUT / "selected_guarded_gpu_residual_policy.csv"
    validation_path = OUT / "gpu_pytorch_validation.json"

    residual_metrics = pd.read_csv(residual_metrics_path)
    selected = pd.read_csv(selected_path).iloc[0].to_dict()

    tree = residual_metrics[residual_metrics["model"].eq("tree_ensemble")].iloc[0]
    raw_gpu = residual_metrics[residual_metrics["model"].eq("torch_gpu_residual_corrected")].iloc[0]
    current = residual_metrics[residual_metrics["model"].eq("current_residual_corrected")].iloc[0]

    validation = {}
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))

    summary_rows = [
        {
            "method": "tree_ensemble_baseline",
            "wape": float(tree["wape"]),
            "rmse": float(tree["rmse"]),
            "mean_bias": float(tree["mean_bias"]),
            "under_forecast_rate": float(tree["under_forecast_rate"]),
            "relative_wape_improvement_vs_tree_pct": 0.0,
            "notes": "CPU tree-ensemble reference forecast.",
        },
        {
            "method": "current_residual_corrected",
            "wape": float(current["wape"]),
            "rmse": float(current["rmse"]),
            "mean_bias": float(current["mean_bias"]),
            "under_forecast_rate": float(current["under_forecast_rate"]),
            "relative_wape_improvement_vs_tree_pct": pct_improvement(float(tree["wape"]), float(current["wape"])),
            "notes": "Existing CPU residual correction artifact.",
        },
        {
            "method": "raw_gpu_residual_corrected",
            "wape": float(raw_gpu["wape"]),
            "rmse": float(raw_gpu["rmse"]),
            "mean_bias": float(raw_gpu["mean_bias"]),
            "under_forecast_rate": float(raw_gpu["under_forecast_rate"]),
            "relative_wape_improvement_vs_tree_pct": pct_improvement(float(tree["wape"]), float(raw_gpu["wape"])),
            "notes": "CUDA-backed PyTorch residual corrector before guardrails.",
        },
        {
            "method": "guarded_gpu_residual_corrected",
            "wape": float(selected["wape"]),
            "rmse": float(selected["rmse"]),
            "mean_bias": float(selected["mean_bias"]),
            "under_forecast_rate": float(selected["under_forecast_rate"]),
            "relative_wape_improvement_vs_tree_pct": pct_improvement(float(tree["wape"]), float(selected["wape"])),
            "lambda": float(selected.get("lambda", 0.0)),
            "clip_quantile": float(selected.get("clip_quantile", 0.0)),
            "notes": "Guarded GPU residual correction using residual clipping and correction-strength scaling.",
        },
    ]

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT / "gpu_specialty_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# GPU Specialty Residual Correction Report",
        "",
        "## Purpose",
        "",
        "This report consolidates the optional CUDA-backed PyTorch residual-correction experiment. The core forecasting pipeline remains CPU-runnable, while this GPU extension evaluates whether a learned residual corrector can improve the tree-ensemble baseline.",
        "",
        "## GPU validation",
        "",
        f"- CUDA available: `{validation.get('cuda_available', 'unknown')}`",
        f"- PyTorch version: `{validation.get('torch_version', 'unknown')}`",
        f"- PyTorch CUDA runtime: `{validation.get('torch_cuda_version', 'unknown')}`",
        f"- Device: `{validation.get('device_name', 'unknown')}`",
        "",
        "## Consolidated metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Raw GPU residual correction improved WAPE from `{float(tree['wape']):.6f}` to `{float(raw_gpu['wape']):.6f}`, a `{pct_improvement(float(tree['wape']), float(raw_gpu['wape'])):.2f}%` relative improvement versus the tree baseline.",
        f"- Guarded GPU residual correction selected `{selected.get('model')}` and preserved WAPE improvement at `{float(selected['wape']):.6f}`, a `{pct_improvement(float(tree['wape']), float(selected['wape'])):.2f}%` relative improvement versus the tree baseline.",
        f"- The guarded version reduced mean bias from `{float(tree['mean_bias']):.6f}` for the tree baseline to `{float(selected['mean_bias']):.6f}`, while avoiding the stronger negative-bias shift of raw GPU correction.",
        "",
        "## Claim boundary",
        "",
        "This is an optional GPU specialty experiment built on generated forecast artifacts. It is useful as GPU ML feasibility evidence and specialty-method evidence. Production-style or resume-critical claims should be made after integrating the guarded GPU residual corrector directly into the main leakage-safe train/validation/test pipeline.",
    ]

    report_path = REP / "gpu_specialty_contribution_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
