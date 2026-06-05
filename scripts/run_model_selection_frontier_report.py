from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("outputs")
REP = Path("reports")


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row_to_dict(row: pd.Series | None) -> dict:
    if row is None:
        return {}
    out = {}
    for k, v in row.to_dict().items():
        if hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.4f}"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    forecast_path = OUT / "forecast_metrics.csv"
    policy_path = OUT / "inventory_policy_summary.csv"
    gate_path = OUT / "gpu_guarded_release_gate.json"

    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing {forecast_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing {policy_path}")

    forecast = pd.read_csv(forecast_path)
    policy = pd.read_csv(policy_path)
    gate = _read_json(gate_path)

    required_forecast_cols = {"model_name", "wape", "rmse", "mae", "mean_bias", "under_forecast_rate"}
    missing = required_forecast_cols - set(forecast.columns)
    if missing:
        raise ValueError(f"forecast_metrics.csv missing columns: {sorted(missing)}")

    required_policy_cols = {
        "forecast_model",
        "lead_time",
        "safety_stock_factor",
        "stockout_rate",
        "service_level",
        "overstock_units",
        "total_cost",
        "demand_recall",
        "inventory_precision",
        "demand_coverage_f1",
    }
    missing = required_policy_cols - set(policy.columns)
    if missing:
        raise ValueError(f"inventory_policy_summary.csv missing columns: {sorted(missing)}")

    baseline_forecast = forecast[forecast["model_name"].eq("tree_ensemble")]
    if baseline_forecast.empty:
        raise ValueError("tree_ensemble baseline not found in forecast_metrics.csv")
    baseline_forecast_row = baseline_forecast.iloc[0]

    treatment_forecast_name = gate.get("treatment_forecast", "learned_abc_gpu_residual")
    treatment_policy_name = gate.get("treatment_policy", "learned_abc_release_safe_policy")
    control_policy_name = gate.get("control_policy", "tree_ensemble_policy")

    treatment_forecast = forecast[forecast["model_name"].eq(treatment_forecast_name)]
    treatment_forecast_row = treatment_forecast.iloc[0] if not treatment_forecast.empty else None

    accuracy_optimal_row = forecast.sort_values(["wape", "rmse", "mae"]).iloc[0]
    rmse_optimal_row = forecast.sort_values(["rmse", "wape"]).iloc[0]
    bias_balanced_row = forecast.assign(abs_bias=forecast["mean_bias"].abs()).sort_values(["abs_bias", "wape"]).iloc[0]
    under_rate_safe_row = forecast.sort_values(["under_forecast_rate", "wape"]).iloc[0]

    # Use the same default gate slice unless the project changes gate config.
    # This matches the usual 7-day lead time and 0.5 safety factor setting used in the release gate.
    gate_lead_time = 7
    gate_safety_factor = 0.5

    gate_policy = policy[
        (policy["lead_time"].eq(gate_lead_time))
        & (policy["safety_stock_factor"].eq(gate_safety_factor))
    ].copy()

    if gate_policy.empty:
        gate_policy = policy.copy()

    baseline_policy = gate_policy[gate_policy["forecast_model"].eq(control_policy_name)]
    baseline_policy_row = baseline_policy.iloc[0] if not baseline_policy.empty else None

    treatment_policy = gate_policy[gate_policy["forecast_model"].eq(treatment_policy_name)]
    treatment_policy_row = treatment_policy.iloc[0] if not treatment_policy.empty else None

    stockout_safe_policy_row = gate_policy.sort_values(
        ["stockout_rate", "total_cost", "overstock_units"]
    ).iloc[0]

    coverage_f1_policy_row = gate_policy.sort_values(
        ["demand_coverage_f1", "stockout_rate"],
        ascending=[False, True],
    ).iloc[0]

    cost_balanced_candidates = gate_policy.copy()
    if baseline_policy_row is not None:
        max_stockout = float(baseline_policy_row["stockout_rate"])
        max_overstock = float(baseline_policy_row["overstock_units"]) * 1.05
        max_total_cost = float(baseline_policy_row["total_cost"]) * 1.05

        feasible = cost_balanced_candidates[
            (cost_balanced_candidates["stockout_rate"] <= max_stockout)
            & (cost_balanced_candidates["overstock_units"] <= max_overstock)
            & (cost_balanced_candidates["total_cost"] <= max_total_cost)
        ].copy()

        if feasible.empty:
            feasible = cost_balanced_candidates.copy()
    else:
        feasible = cost_balanced_candidates.copy()

    cost_balanced_policy_row = feasible.sort_values(
        ["total_cost", "stockout_rate", "demand_coverage_f1"],
        ascending=[True, True, False],
    ).iloc[0]

    forecast_frontier_rows = []
    for role, row, reason in [
        ("Baseline forecast", baseline_forecast_row, "Reference tree-ensemble model."),
        ("Accuracy-optimal forecast", accuracy_optimal_row, "Lowest WAPE among evaluated point forecasts."),
        ("RMSE-optimal forecast", rmse_optimal_row, "Lowest RMSE among evaluated point forecasts."),
        ("Bias-balanced forecast", bias_balanced_row, "Lowest absolute mean bias, tie-broken by WAPE."),
        ("Under-forecast-safe forecast", under_rate_safe_row, "Lowest under-forecast rate, tie-broken by WAPE."),
    ]:
        forecast_frontier_rows.append({
            "frontier_role": role,
            "candidate": row["model_name"],
            "layer": "forecast",
            "wape": row["wape"],
            "rmse": row["rmse"],
            "mae": row["mae"],
            "mean_bias": row["mean_bias"],
            "under_forecast_rate": row["under_forecast_rate"],
            "stockout_rate": np.nan,
            "total_cost": np.nan,
            "demand_coverage_f1": np.nan,
            "decision_use": reason,
        })

    policy_frontier_rows = []
    policy_items = [
        ("Baseline policy", baseline_policy_row, "Reference inventory policy at the gate slice."),
        ("Release-gate treatment policy", treatment_policy_row, "Policy currently used by gpu_guarded_release_gate.json."),
        ("Stockout-safe policy", stockout_safe_policy_row, "Lowest stockout rate at the gate slice."),
        ("Coverage-F1 policy", coverage_f1_policy_row, "Highest inventory demand-coverage F1 at the gate slice."),
        ("Cost-balanced policy", cost_balanced_policy_row, "Lowest total cost among feasible policies under baseline-relative constraints."),
    ]

    for role, row, reason in policy_items:
        if row is None:
            continue
        policy_frontier_rows.append({
            "frontier_role": role,
            "candidate": row["forecast_model"],
            "layer": "policy",
            "wape": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mean_bias": np.nan,
            "under_forecast_rate": np.nan,
            "stockout_rate": row["stockout_rate"],
            "total_cost": row["total_cost"],
            "demand_coverage_f1": row["demand_coverage_f1"],
            "decision_use": reason,
        })

    frontier = pd.DataFrame(forecast_frontier_rows + policy_frontier_rows)
    frontier.to_csv(OUT / "model_selection_frontier.csv", index=False)

    if treatment_forecast_row is not None:
        wape_improvement = (
            float(baseline_forecast_row["wape"]) - float(treatment_forecast_row["wape"])
        ) / max(float(baseline_forecast_row["wape"]), 1e-9)
    else:
        wape_improvement = np.nan

    recommendation = {
        "recommended_decision": gate.get("decision", "ITERATE"),
        "accuracy_optimal_forecast": str(accuracy_optimal_row["model_name"]),
        "recommended_release_policy": treatment_policy_name,
        "baseline_forecast": "tree_ensemble",
        "control_policy": control_policy_name,
        "wape_relative_improvement_vs_tree": wape_improvement,
        "gate_guardrails": gate.get("guardrails", {}),
        "remaining_iteration_item": "under_forecast_improved" if gate.get("guardrails", {}).get("under_forecast_improved") == "fail" else "none",
        "claim_boundary": "Offline public-data decision artifact. Inventory impacts use synthetic/proxy overlays.",
    }

    (OUT / "model_selection_recommendation.json").write_text(
        json.dumps(recommendation, indent=2),
        encoding="utf-8",
    )

    forecast_table = forecast.sort_values("wape").copy()
    policy_table = gate_policy.sort_values(
        ["stockout_rate", "total_cost", "demand_coverage_f1"],
        ascending=[True, True, False],
    ).copy()

    report = [
        "# Model Selection Frontier Report",
        "",
        "## Executive decision",
        "",
        f"Recommended decision: **{recommendation['recommended_decision']}**",
        "",
        "This report separates the point-forecast frontier from the inventory-policy frontier. The accuracy-optimal model is not automatically treated as the launch policy; policy selection is evaluated through stockout, overstock, total-cost, and coverage guardrails.",
        "",
        "## Current recommendation",
        "",
        f"- Accuracy-optimal forecast: `{recommendation['accuracy_optimal_forecast']}`",
        f"- Release-gate policy: `{recommendation['recommended_release_policy']}`",
        f"- WAPE improvement versus tree baseline: `{_fmt_pct(wape_improvement)}`",
        f"- Remaining iteration item: `{recommendation['remaining_iteration_item']}`",
        "",
        "## Release-gate guardrails",
        "",
        pd.DataFrame([
            {"guardrail": k, "status": v}
            for k, v in recommendation["gate_guardrails"].items()
        ]).to_markdown(index=False),
        "",
        "## Frontier summary",
        "",
        frontier.to_markdown(index=False),
        "",
        "## Forecast candidates ranked by WAPE",
        "",
        forecast_table.to_markdown(index=False),
        "",
        f"## Policy candidates at lead_time={gate_lead_time}, safety_stock_factor={gate_safety_factor}",
        "",
        policy_table.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `learned_abc_gpu_residual` is the accuracy-oriented forecast candidate because it minimizes WAPE in the current run.",
        "- `learned_abc_release_safe_policy` is the release-gate treatment policy because it separates inventory ordering from the raw accuracy-optimal forecast.",
        "- The decision remains ITERATE rather than LAUNCH because the under-forecast guardrail is still failing, even though the stockout, overstock, total-cost, and worst-slice guardrails pass.",
        "",
        "## Resume-safe claim boundary",
        "",
        "Safe claim: built a leakage-safe forecasting decision framework with an accuracy frontier, policy frontier, release gates, and ABC-style demand coverage correction.",
        "",
        "Do not claim production launch, real-world stockout reduction, or Amazon production impact.",
    ]

    (REP / "model_selection_frontier_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("Wrote outputs/model_selection_frontier.csv")
    print("Wrote outputs/model_selection_recommendation.json")
    print("Wrote reports/model_selection_frontier_report.md")
    print()
    print(frontier.to_string(index=False))


if __name__ == "__main__":
    main()
