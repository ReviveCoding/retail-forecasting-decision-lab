from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs")
REP = Path("reports")
SOURCE = OUT / "corrected_forecasts_sample.csv"

CANDIDATES = [
    "pred_seasonal_naive",
    "pred_moving_average",
    "pred_tree",
    "pred_corrected",
    "pred_guarded_gpu_residual",
    "pred_release_safe_gpu_residual",
    "order_signal_guarded_gpu_residual",
    "order_signal_release_safe_gpu_residual",
]


def demand_coverage_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    valid = df[["demand", pred_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = valid["demand"].astype(float).to_numpy()
    p = valid[pred_col].astype(float).clip(lower=0).to_numpy()

    covered = np.minimum(y, p)
    under = np.maximum(y - p, 0.0)
    over = np.maximum(p - y, 0.0)

    actual_sum = max(float(y.sum()), 1e-9)
    pred_sum = max(float(p.sum()), 1e-9)

    coverage_recall = float(covered.sum() / actual_sum)
    order_precision = float(covered.sum() / pred_sum)
    coverage_f1 = float(
        2.0 * coverage_recall * order_precision
        / max(coverage_recall + order_precision, 1e-9)
    )

    wape = float(np.abs(y - p).sum() / actual_sum)

    return {
        "candidate": pred_col,
        "n_rows": int(len(valid)),
        "actual_sum": float(y.sum()),
        "pred_sum": float(p.sum()),
        "wape": wape,
        "demand_coverage_recall": coverage_recall,
        "order_precision": order_precision,
        "demand_coverage_f1": coverage_f1,
        "under_coverage_amount": float(under.sum()),
        "over_coverage_amount": float(over.sum()),
        "under_coverage_rate": float((p < y).mean()),
        "over_coverage_rate": float((p > y).mean()),
        "mean_bias": float((p - y).mean()),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    df = pd.read_csv(SOURCE)
    available = [c for c in CANDIDATES if c in df.columns]

    if not available:
        raise ValueError("No candidate prediction columns found.")

    rows = [demand_coverage_metrics(df, c) for c in available]
    metrics = pd.DataFrame(rows)

    tree = metrics.loc[metrics["candidate"].eq("pred_tree")]
    if not tree.empty:
        tree_row = tree.iloc[0]
        metrics["coverage_recall_delta_vs_tree"] = (
            metrics["demand_coverage_recall"] - float(tree_row["demand_coverage_recall"])
        )
        metrics["order_precision_delta_vs_tree"] = (
            metrics["order_precision"] - float(tree_row["order_precision"])
        )
        metrics["coverage_f1_delta_vs_tree"] = (
            metrics["demand_coverage_f1"] - float(tree_row["demand_coverage_f1"])
        )
        metrics["wape_delta_vs_tree"] = metrics["wape"] - float(tree_row["wape"])

    metrics = metrics.sort_values(
        ["demand_coverage_f1", "demand_coverage_recall", "order_precision"],
        ascending=False,
    )

    metrics.to_csv(OUT / "demand_coverage_metrics.csv", index=False)

    slice_cols = [c for c in ["cat_id", "dept_id", "store_id", "state_id"] if c in df.columns]
    slice_rows = []

    for pred_col in available:
        for slice_col in slice_cols:
            for value, g in df.groupby(slice_col):
                if len(g) < 10:
                    continue
                row = demand_coverage_metrics(g, pred_col)
                row["slice_col"] = slice_col
                row["slice_value"] = value
                slice_rows.append(row)

    if slice_rows:
        slice_df = pd.DataFrame(slice_rows)
        slice_df.to_csv(OUT / "demand_coverage_by_slice.csv", index=False)

    report = [
        "# ABC-Style Demand Coverage Reliability Report",
        "",
        "## Purpose",
        "",
        "This report adapts the ABC Bbox region-wise Recall, Precision, and F1 idea to retail demand forecasting.",
        "",
        "Demand coverage recall measures how much actual demand is covered by a forecast or order signal.",
        "",
        "Order precision measures how much forecasted or ordered demand is justified by actual demand.",
        "",
        "Demand coverage F1 summarizes the coverage-risk trade-off between under-forecasting and over-forecasting.",
        "",
        "## Source",
        "",
        f"`{SOURCE}`",
        "",
        "## Candidate metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Interpretation guide",
        "",
        "- Higher demand coverage recall means lower under-coverage risk.",
        "- Higher order precision means lower over-coverage risk.",
        "- Higher demand coverage F1 means a better balance between coverage and precision.",
        "- A candidate can improve WAPE while still hurting coverage risk, so this report should be read together with forecast_metrics.csv and release_gate.json.",
        "",
        "## Claim boundary",
        "",
        "This is an offline public-data evaluation artifact, not evidence of production inventory impact.",
    ]

    (REP / "demand_coverage_reliability_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("Wrote outputs/demand_coverage_metrics.csv")
    print("Wrote outputs/demand_coverage_by_slice.csv")
    print("Wrote reports/demand_coverage_reliability_report.md")
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
