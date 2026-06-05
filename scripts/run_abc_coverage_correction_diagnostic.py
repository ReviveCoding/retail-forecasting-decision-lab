from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs")
REP = Path("reports")
SOURCE = OUT / "corrected_forecasts_sample.csv"


def _minmax_from_guard(x_guard: pd.Series, x_apply: pd.Series) -> np.ndarray:
    lo = float(np.nanpercentile(x_guard.astype(float), 5))
    hi = float(np.nanpercentile(x_guard.astype(float), 95))
    if hi <= lo:
        return np.zeros(len(x_apply), dtype=float)
    z = (x_apply.astype(float).to_numpy() - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)


def add_under_risk_score(guard: pd.DataFrame, apply: pd.DataFrame) -> np.ndarray:
    parts = []

    for col, weight in [
        ("model_disagreement_score", 0.25),
        ("candidate_forecast_std", 0.25),
        ("rolling_std_28", 0.15),
        ("rolling_std_7", 0.10),
        ("volatility_state", 0.10),
    ]:
        if col in guard.columns and col in apply.columns:
            parts.append(weight * _minmax_from_guard(guard[col], apply[col]))

    for col, weight in [
        ("event_flag", 0.075),
        ("snap_flag", 0.075),
    ]:
        if col in apply.columns:
            parts.append(weight * apply[col].fillna(0).astype(float).clip(0, 1).to_numpy())

    if not parts:
        return np.ones(len(apply), dtype=float)

    risk = np.sum(parts, axis=0)
    max_risk = max(float(np.nanmax(risk)), 1e-9)
    return np.clip(risk / max_risk, 0.0, 1.0)


def metrics(df: pd.DataFrame, pred: np.ndarray, name: str) -> dict:
    y = df["demand"].astype(float).to_numpy()
    p = np.clip(pred.astype(float), 0.0, None)

    covered = np.minimum(y, p)
    actual_sum = max(float(y.sum()), 1e-9)
    pred_sum = max(float(p.sum()), 1e-9)

    recall = float(covered.sum() / actual_sum)
    precision = float(covered.sum() / pred_sum)
    f1 = float(2 * recall * precision / max(recall + precision, 1e-9))

    return {
        "candidate": name,
        "n_rows": int(len(df)),
        "actual_sum": float(y.sum()),
        "pred_sum": float(p.sum()),
        "wape": float(np.abs(y - p).sum() / actual_sum),
        "demand_coverage_recall": recall,
        "order_precision": precision,
        "demand_coverage_f1": f1,
        "under_coverage_amount": float(np.maximum(y - p, 0.0).sum()),
        "over_coverage_amount": float(np.maximum(p - y, 0.0).sum()),
        "under_coverage_rate": float((p < y).mean()),
        "over_coverage_rate": float((p > y).mean()),
        "mean_bias": float((p - y).mean()),
    }


def choose_candidate(grid: pd.DataFrame, tree: dict, guarded: dict) -> pd.Series:
    candidates = grid.copy()

    # Prefer candidates that retain better-than-tree WAPE and improve under-rate vs guarded.
    feasible = candidates[
        (candidates["wape"] <= tree["wape"])
        & (candidates["under_coverage_rate"] <= guarded["under_coverage_rate"])
    ].copy()

    if feasible.empty:
        feasible = candidates.copy()

    feasible["score"] = (
        feasible["wape"].rank(method="min")
        + feasible["under_coverage_rate"].rank(method="min")
        + feasible["mean_bias"].abs().rank(method="min")
        + (-feasible["demand_coverage_f1"]).rank(method="min")
    )

    return feasible.sort_values(["score", "wape"]).iloc[0]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}")

    df = pd.read_csv(SOURCE)
    required = [
        "date",
        "item_store_id",
        "demand",
        "pred_tree",
        "pred_guarded_gpu_residual",
        "pred_release_safe_gpu_residual",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "item_store_id"]).reset_index(drop=True)

    unique_dates = sorted(df["date"].unique())
    cut_date = unique_dates[len(unique_dates) // 2]

    guard = df[df["date"] <= cut_date].copy()
    eval_df = df[df["date"] > cut_date].copy()

    if len(guard) < 100 or len(eval_df) < 100:
        raise ValueError("Not enough rows for diagnostic guard/eval split.")

    guard_risk = add_under_risk_score(guard, guard)
    eval_risk = add_under_risk_score(guard, eval_df)

    guard_base = guard["pred_guarded_gpu_residual"].astype(float).to_numpy()
    guard_floor = guard["pred_release_safe_gpu_residual"].astype(float).to_numpy()
    guard_gap = np.maximum(guard_floor - guard_base, 0.0)

    eval_base = eval_df["pred_guarded_gpu_residual"].astype(float).to_numpy()
    eval_floor = eval_df["pred_release_safe_gpu_residual"].astype(float).to_numpy()
    eval_gap = np.maximum(eval_floor - eval_base, 0.0)

    tree_guard = metrics(guard, guard["pred_tree"].to_numpy(), "pred_tree_guard")
    guarded_guard = metrics(guard, guard_base, "pred_guarded_gpu_guard")
    release_guard = metrics(guard, guard_floor, "pred_release_safe_gpu_guard")

    rows = []
    for alpha in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00]:
        for risk_power in [0.5, 1.0, 2.0]:
            for q in [0.0, 0.25, 0.50, 0.75]:
                threshold = float(np.quantile(guard_risk, q))
                weight = np.where(guard_risk >= threshold, guard_risk ** risk_power, 0.0)
                pred = guard_base + float(alpha) * weight * guard_gap
                row = metrics(guard, pred, "abc_guard_grid")
                row["alpha"] = float(alpha)
                row["risk_power"] = float(risk_power)
                row["risk_quantile_gate"] = float(q)
                row["risk_threshold"] = threshold
                rows.append(row)

    grid = pd.DataFrame(rows)
    selected = choose_candidate(grid, tree_guard, guarded_guard)

    alpha = float(selected["alpha"])
    risk_power = float(selected["risk_power"])
    q = float(selected["risk_quantile_gate"])
    threshold = float(selected["risk_threshold"])

    eval_weight = np.where(eval_risk >= threshold, eval_risk ** risk_power, 0.0)
    eval_pred = eval_base + alpha * eval_weight * eval_gap

    eval_rows = [
        metrics(eval_df, eval_df["pred_tree"].to_numpy(), "pred_tree_eval"),
        metrics(eval_df, eval_base, "pred_guarded_gpu_eval"),
        metrics(eval_df, eval_floor, "pred_release_safe_gpu_eval"),
        metrics(eval_df, eval_pred, "pred_abc_selective_gpu_eval"),
    ]
    eval_metrics = pd.DataFrame(eval_rows)

    grid.to_csv(OUT / "abc_coverage_diagnostic_grid.csv", index=False)
    pd.DataFrame([selected.to_dict()]).to_csv(
        OUT / "abc_coverage_diagnostic_selected.csv",
        index=False,
    )
    eval_metrics.to_csv(OUT / "abc_coverage_diagnostic_eval.csv", index=False)

    report = [
        "# ABC-Style Selective Demand Coverage Correction Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether ABC-style selective demand inflation can improve the trade-off between WAPE, demand coverage recall, and order precision.",
        "",
        "The script does not modify the main pipeline. It splits the generated holdout artifact into a diagnostic guard half and diagnostic evaluation half.",
        "",
        "## Selected guard parameters",
        "",
        pd.DataFrame([selected.to_dict()]).to_markdown(index=False),
        "",
        "## Diagnostic evaluation metrics",
        "",
        eval_metrics.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `pred_guarded_gpu_eval` is the WAPE-oriented GPU forecast.",
        "- `pred_release_safe_gpu_eval` is the conservative floor-based forecast.",
        "- `pred_abc_selective_gpu_eval` selectively moves the guarded forecast toward the conservative forecast only for high under-risk rows.",
        "",
        "## Claim boundary",
        "",
        "This is a diagnostic experiment. Do not use it as a resume performance claim until the same logic is integrated into the leakage-safe train-base / residual-fit / guard-validation / test pipeline.",
    ]

    (REP / "abc_coverage_correction_diagnostic_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("Wrote outputs/abc_coverage_diagnostic_grid.csv")
    print("Wrote outputs/abc_coverage_diagnostic_selected.csv")
    print("Wrote outputs/abc_coverage_diagnostic_eval.csv")
    print("Wrote reports/abc_coverage_correction_diagnostic_report.md")
    print()
    print(eval_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
