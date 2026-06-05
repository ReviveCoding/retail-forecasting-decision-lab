from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


OUT = Path("outputs")
REP = Path("reports")
SOURCE = OUT / "corrected_forecasts_sample.csv"


FEATURE_CANDIDATES = [
    "pred_tree",
    "pred_corrected",
    "pred_guarded_gpu_residual",
    "pred_release_safe_gpu_residual",
    "candidate_forecast_mean",
    "candidate_forecast_std",
    "model_disagreement_score",
    "pred_p10",
    "pred_p90",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "sell_price",
    "price_lag_7",
    "price_change_7",
    "cat_demand_lag_mean",
    "dept_demand_lag_mean",
    "store_demand_lag_mean",
    "state_demand_lag_mean",
    "normalized_lifecycle_age",
    "rolling_demand_state",
    "volatility_state",
    "intermittency_score",
    "event_recent_7",
    "category_relative_age",
    "zero_demand_recent_7",
    "demand_drop_flag",
    "event_flag",
    "snap_flag",
    "dayofweek",
    "weekofyear",
    "is_weekend",
]


def metric_row(df: pd.DataFrame, pred: np.ndarray, name: str) -> dict:
    y = df["demand"].astype(float).to_numpy()
    p = np.clip(pred.astype(float), 0.0, None)

    covered = np.minimum(y, p)
    actual_sum = max(float(y.sum()), 1e-9)
    pred_sum = max(float(p.sum()), 1e-9)

    recall = float(covered.sum() / actual_sum)
    precision = float(covered.sum() / pred_sum)
    f1 = float(2.0 * recall * precision / max(recall + precision, 1e-9))

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


def make_xy(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return x.astype(float).to_numpy()


def choose_candidate(grid: pd.DataFrame, tree: dict, guarded: dict) -> pd.Series:
    candidates = grid.copy()

    feasible = candidates[
        (candidates["wape"] <= tree["wape"])
        & (candidates["under_coverage_rate"] <= guarded["under_coverage_rate"])
        & (candidates["demand_coverage_f1"] >= guarded["demand_coverage_f1"])
    ].copy()

    if feasible.empty:
        feasible = candidates.copy()

    feasible["score"] = (
        feasible["wape"].rank(method="min")
        + feasible["under_coverage_rate"].rank(method="min")
        + (-feasible["demand_coverage_f1"]).rank(method="min")
        + feasible["mean_bias"].abs().rank(method="min")
    )

    return feasible.sort_values(["score", "wape"]).iloc[0]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    df = pd.read_csv(SOURCE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "item_store_id"]).reset_index(drop=True)

    required = [
        "demand",
        "pred_tree",
        "pred_guarded_gpu_residual",
        "pred_release_safe_gpu_residual",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)

    fit_cut = unique_dates[int(n_dates * 0.40)]
    select_cut = unique_dates[int(n_dates * 0.65)]

    fit_df = df[df["date"] <= fit_cut].copy()
    select_df = df[(df["date"] > fit_cut) & (df["date"] <= select_cut)].copy()
    eval_df = df[df["date"] > select_cut].copy()

    if len(fit_df) < 100 or len(select_df) < 100 or len(eval_df) < 100:
        raise ValueError("Not enough rows for fit/select/eval split.")

    y_fit = fit_df["demand"].astype(float).to_numpy()
    guarded_fit = fit_df["pred_guarded_gpu_residual"].astype(float).to_numpy()

    under_target = np.maximum(y_fit - guarded_fit, 0.0)
    over_target = np.maximum(guarded_fit - y_fit, 0.0)

    x_fit = make_xy(fit_df, feature_cols)
    x_select = make_xy(select_df, feature_cols)
    x_eval = make_xy(eval_df, feature_cols)

    under_model = HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=42,
    )
    over_model = HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=43,
    )

    under_model.fit(x_fit, under_target)
    over_model.fit(x_fit, over_target)

    select_under = np.clip(under_model.predict(x_select), 0.0, None)
    select_over = np.clip(over_model.predict(x_select), 0.0, None)

    eval_under = np.clip(under_model.predict(x_eval), 0.0, None)
    eval_over = np.clip(over_model.predict(x_eval), 0.0, None)

    select_base = select_df["pred_guarded_gpu_residual"].astype(float).to_numpy()
    eval_base = eval_df["pred_guarded_gpu_residual"].astype(float).to_numpy()

    tree_select = metric_row(
        select_df,
        select_df["pred_tree"].to_numpy(),
        "pred_tree_select",
    )
    guarded_select = metric_row(
        select_df,
        select_base,
        "pred_guarded_gpu_select",
    )

    grid_rows = []

    for alpha in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]:
        for beta in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
            for cap_q in [0.75, 0.90, 0.95, 1.00]:
                under_cap = float(np.quantile(select_under, cap_q))
                over_cap = float(np.quantile(select_over, cap_q))

                pred = (
                    select_base
                    + float(alpha) * np.minimum(select_under, under_cap)
                    - float(beta) * np.minimum(select_over, over_cap)
                )

                row = metric_row(select_df, pred, "learned_abc_select_grid")
                row["alpha"] = float(alpha)
                row["beta"] = float(beta)
                row["cap_quantile"] = float(cap_q)
                row["under_cap"] = under_cap
                row["over_cap"] = over_cap
                grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    selected = choose_candidate(grid, tree_select, guarded_select)

    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    under_cap = float(selected["under_cap"])
    over_cap = float(selected["over_cap"])

    eval_pred = (
        eval_base
        + alpha * np.minimum(eval_under, under_cap)
        - beta * np.minimum(eval_over, over_cap)
    )

    eval_metrics = pd.DataFrame(
        [
            metric_row(
                eval_df,
                eval_df["pred_tree"].to_numpy(),
                "pred_tree_eval",
            ),
            metric_row(
                eval_df,
                eval_df["pred_guarded_gpu_residual"].to_numpy(),
                "pred_guarded_gpu_eval",
            ),
            metric_row(
                eval_df,
                eval_df["pred_release_safe_gpu_residual"].to_numpy(),
                "pred_release_safe_gpu_eval",
            ),
            metric_row(
                eval_df,
                eval_pred,
                "pred_learned_abc_gpu_eval",
            ),
        ]
    )

    grid.to_csv(OUT / "learned_abc_coverage_grid.csv", index=False)
    pd.DataFrame([selected.to_dict()]).to_csv(
        OUT / "learned_abc_coverage_selected.csv",
        index=False,
    )
    eval_metrics.to_csv(OUT / "learned_abc_coverage_eval.csv", index=False)

    report = [
        "# Learned ABC-Style Demand Coverage Correction Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic learns under-coverage and over-coverage targets from the generated forecasting artifact, then evaluates whether a learned ABC-style asymmetric correction improves the WAPE and coverage trade-off.",
        "",
        "## Split",
        "",
        f"- fit rows: {len(fit_df)}",
        f"- selection rows: {len(select_df)}",
        f"- evaluation rows: {len(eval_df)}",
        "",
        "## Selected parameters",
        "",
        pd.DataFrame([selected.to_dict()]).to_markdown(index=False),
        "",
        "## Evaluation metrics",
        "",
        eval_metrics.to_markdown(index=False),
        "",
        "## Claim boundary",
        "",
        "This is a diagnostic experiment on generated holdout artifacts. It should be integrated into the main leakage-safe pipeline before being used as a performance claim.",
    ]

    (REP / "learned_abc_coverage_correction_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("Wrote outputs/learned_abc_coverage_grid.csv")
    print("Wrote outputs/learned_abc_coverage_selected.csv")
    print("Wrote outputs/learned_abc_coverage_eval.csv")
    print("Wrote reports/learned_abc_coverage_correction_report.md")
    print()
    print(eval_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
