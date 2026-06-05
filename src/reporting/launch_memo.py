from __future__ import annotations


def build_launch_memo_text(gate: dict) -> str:
    guardrails = gate.get('guardrails', {})
    lines = [f"## Decision: {gate.get('decision', 'UNKNOWN')}", ""]
    lines.append("This decision compares a tree-ensemble point forecast with a residual-corrected forecast, then evaluates the risk-safe corrected ordering policy under a public-data, offline, synthetic-inventory benchmark.")
    lines.append("")
    lines.append("## Key evidence")
    lines.append(f"- WAPE relative improvement: {gate.get('wape_relative_improvement'):.4f}")
    lines.append(f"- Under-forecast relative improvement: {gate.get('under_forecast_relative_improvement'):.4f}")
    lines.append(f"- Stockout delta, treatment minus control: {gate.get('stockout_delta_treatment_minus_control')}")
    lines.append(f"- Overstock relative worsening: {gate.get('overstock_relative_worsening')}")
    lines.append(f"- Total-cost relative worsening: {gate.get('total_cost_relative_worsening')}")
    lines.append("")
    lines.append("## Guardrails")
    for k, v in guardrails.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Recommendation")
    if gate.get('decision') == 'LAUNCH':
        lines.append("Proceed in the offline benchmark and prepare a larger validation run.")
    elif gate.get('decision') == 'ITERATE':
        lines.append("Continue tuning because some guardrails pass but launch-strength evidence is incomplete.")
    elif gate.get('decision') == 'BLOCK':
        lines.append("Do not launch this treatment policy because a business guardrail failed materially.")
    else:
        lines.append("Hold the release. The benchmark shows useful trade-off evidence, but not enough to clear the decision gate.")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("This is not a production Amazon launch memo. It is a public-data, local-runnable measurement artifact.")
    return "\n".join(lines)
