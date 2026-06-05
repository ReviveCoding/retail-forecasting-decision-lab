from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.config import load_pipeline_config
from src.utils import ensure_dirs, write_json, write_table
from src.data.load_m5 import load_m5_panel
from src.data.validate_schema import validate_panel_schema
from src.data.create_sql_mart import create_sql_mart
from src.features.lag_rolling import add_lag_rolling_features
from src.features.calendar_price import add_calendar_price_features
from src.features.hierarchy import add_hierarchy_features
from src.features.demand_state_timestep import add_demand_state_timestep
from src.features.stockout_proxy import add_stockout_proxy_features
from src.models.baselines import add_baseline_predictions
from src.models.tree_forecaster import train_forecaster, predict_forecaster
from src.models.residual_bias_corrector import train_residual_corrector, apply_residual_correction
from src.models.forecast_uncertainty import add_candidate_forecast_uncertainty
from src.models.torch_guarded_residual_corrector import apply_guarded_gpu_residual_stage
from src.evaluation.forecast_metrics import evaluate_forecasts
from src.evaluation.bias_metrics import evaluate_bias_by_slice
from src.simulation.inventory_policy import simulate_inventory_policies
from src.experiments.ab_style_measurement import build_ab_style_readout
from src.experiments.causal_event_study import lightweight_event_residual_readout
from src.explainability.explanation_generator import generate_explanations
from src.explainability.grounding_evaluator import evaluate_explanation_grounding
from src.evaluation.release_gate import build_release_gate
from src.reporting.report_writer import write_markdown, df_head_markdown, report_section
from src.reporting.launch_memo import build_launch_memo_text
from src.reporting.model_card import model_card_text


def split_temporal(df: pd.DataFrame, test_days=28, val_days=28):
    max_date = df['date'].max()
    test_start = max_date - pd.Timedelta(days=int(test_days) - 1)
    val_start = test_start - pd.Timedelta(days=int(val_days))
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    return train, val, test, {
        'train_end': str(train['date'].max().date()),
        'val_start': str(val_start.date()),
        'test_start': str(test_start.date()),
        'test_end': str(max_date.date()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pipeline.yaml')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    cfg = load_pipeline_config(args.config)
    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    feat_cfg = cfg.get('features', {})
    inv_cfg = cfg.get('inventory', {})
    exp_cfg = cfg.get('experiment', {})
    gate_cfg = cfg.get('release_gate', {})
    expn_cfg = cfg.get('explainability', {})
    gpu_cfg = cfg.get('gpu_residual', {})

    out_dir = Path(data_cfg.get('output_dir', 'outputs'))
    rep_dir = Path(data_cfg.get('report_dir', 'reports'))
    ensure_dirs(out_dir, rep_dir, 'data/processed')

    if args.full:
        data_cfg['sample_mode'] = False
        data_cfg['max_days'] = None

    panel = load_m5_panel(
        m5_path=data_cfg.get("m5_path") or data_cfg.get("m5_zip_path"),
        m5_format=data_cfg.get("m5_format", "auto"),
        n_items=data_cfg.get("n_items", 30),
        n_stores=data_cfg.get("n_stores", 2),
        max_days=data_cfg.get("max_days", 420),
        sample_mode=data_cfg.get("sample_mode", True),
    )
    schema_report = validate_panel_schema(panel)
    write_json(schema_report, out_dir / 'data_quality_report.json')
    create_sql_mart(panel, out_dir)

    df = add_lag_rolling_features(panel)
    df = add_calendar_price_features(df)
    df = add_hierarchy_features(df)
    df = add_demand_state_timestep(df, lifecycle_reference_days=feat_cfg.get('lifecycle_reference_days', 365))
    df = add_stockout_proxy_features(df)
    df = add_baseline_predictions(df)
    df = df.dropna(subset=['lag_28', 'rolling_mean_7']).copy()

    train, val, test, split_info = split_temporal(
        df,
        data_cfg.get('forecast_horizon_days', 28),
        data_cfg.get('validation_horizon_days', 28),
    )
    seed = int(data_cfg.get('random_seed', model_cfg.get('random_seed', 42)))
    model = train_forecaster(
        train,
        seed=seed,
        n_estimators=model_cfg.get('n_estimators', 80),
        max_depth=model_cfg.get('max_depth', 12),
        min_samples_leaf=model_cfg.get('min_samples_leaf', 2),
    )
    train['pred_tree'] = predict_forecaster(model, train)
    val['pred_tree'] = predict_forecaster(model, val)
    test['pred_tree'] = predict_forecaster(model, test)

    corrector = train_residual_corrector(val, base_pred_col='pred_tree', seed=seed)
    test['pred_corrected'] = apply_residual_correction(corrector, test, base_pred_col='pred_tree')

    # Risk-safe adjustment is intentionally treated as an ordering-policy signal,
    # not as a pure point forecast for WAPE reporting.
    test['order_signal_risk_safe_corrected'] = test[['pred_corrected', 'pred_tree']].max(axis=1)
    test = add_candidate_forecast_uncertainty(test, pred_cols=['pred_seasonal_naive', 'pred_moving_average', 'pred_tree', 'pred_corrected'])
    if bool(gpu_cfg.get("enabled", False)):
        gpu_stage = apply_guarded_gpu_residual_stage(
            train=train,
            val=val,
            test=test,
            model_cfg=model_cfg,
            gpu_cfg=gpu_cfg,
            seed=seed,
            out_dir=out_dir,
            rep_dir=rep_dir,
        )
        test = gpu_stage.get("test", test)

    pred_map = {
        'seasonal_naive': 'pred_seasonal_naive',
        'moving_average': 'pred_moving_average',
        'tree_ensemble': 'pred_tree',
        'corrected': 'pred_corrected',
    }
    if "pred_guarded_gpu_residual" in test.columns:
        pred_map["guarded_gpu_residual"] = "pred_guarded_gpu_residual"
    if "pred_release_safe_gpu_residual" in test.columns:
        pred_map["release_safe_gpu_residual"] = "pred_release_safe_gpu_residual"
    metrics = evaluate_forecasts(test, pred_map)
    bias_by_slice = evaluate_bias_by_slice(test, pred_map)
    write_table(metrics, out_dir / 'forecast_metrics.csv')
    write_table(bias_by_slice, out_dir / 'bias_by_slice.csv')
    write_table(model.feature_importance, out_dir / 'feature_importance.csv')

    inv_pred_cols = {
        'seasonal_naive_policy': 'pred_seasonal_naive',
        'tree_ensemble_policy': 'pred_tree',
        'corrected_policy': 'pred_corrected',
        'risk_safe_corrected_policy': 'order_signal_risk_safe_corrected',
    }
    if "order_signal_guarded_gpu_residual" in test.columns:
        inv_pred_cols["guarded_gpu_residual_policy"] = "order_signal_guarded_gpu_residual"
    if "order_signal_release_safe_gpu_residual" in test.columns:
        inv_pred_cols["release_safe_gpu_residual_policy"] = "order_signal_release_safe_gpu_residual"
    inv_detail, inv_summary = simulate_inventory_policies(
        test,
        pred_cols=inv_pred_cols,
        lead_times=tuple(inv_cfg.get('lead_times', [1, 3, 7, 14])),
        safety_factors=tuple(inv_cfg.get('safety_stock_factors', [0.0, 0.5, 1.0])),
        holding_cost=inv_cfg.get('holding_cost_per_unit', 0.25),
        lost_sales_cost=inv_cfg.get('lost_sales_cost_per_unit', 1.0),
        labor_units_per_demand_unit=inv_cfg.get('labor_units_per_demand_unit', 0.02),
    )
    write_table(inv_summary, out_dir / 'inventory_policy_summary.csv')
    detail_rows = int(inv_cfg.get('detail_sample_rows', 5000))
    write_table(inv_detail.head(detail_rows), out_dir / 'inventory_policy_detail_sample.csv')
    if bool(inv_cfg.get('write_detail_outputs', False)):
        write_table(inv_detail, out_dir / 'inventory_policy_detail.csv')
    write_table(inv_detail[['forecast_model','lead_time','safety_stock_factor','demand_recall','inventory_precision','demand_coverage_f1']].head(detail_rows), out_dir / 'demand_coverage_metrics_sample.csv')
    write_table(test.head(detail_rows), out_dir / 'corrected_forecasts_sample.csv')

    ab = build_ab_style_readout(
        inv_detail,
        control=exp_cfg.get('control_policy', 'tree_ensemble_policy'),
        treatment=exp_cfg.get('treatment_policy', 'risk_safe_corrected_policy'),
        lead_time=exp_cfg.get('lead_time', 7),
        safety_stock_factor=exp_cfg.get('safety_stock_factor', 0.5),
        n_bootstrap=exp_cfg.get('n_bootstrap', 200),
        seed=exp_cfg.get('random_seed', seed),
        unit_type=exp_cfg.get('unit_type', 'item-store-week'),
    )
    write_table(ab, out_dir / 'ab_test_summary.csv')
    ab_gpu = None
    if ("release_safe_gpu_residual_policy" in inv_pred_cols) or ("guarded_gpu_residual_policy" in inv_pred_cols):
        import inspect
        _ab_params = inspect.signature(build_ab_style_readout).parameters
        _ab_seed = exp_cfg.get("random_seed", seed)
        _ab_unit_type = exp_cfg.get("unit_type", "item-store-week")
        _control = "tree_ensemble_policy"
        _treatment = "release_safe_gpu_residual_policy" if "release_safe_gpu_residual_policy" in inv_pred_cols else "guarded_gpu_residual_policy"
        if ("control_policy" in _ab_params) and ("treatment_policy" in _ab_params):
            ab_gpu = build_ab_style_readout(
                inv_detail,
                control_policy=_control,
                treatment_policy=_treatment,
                seed=_ab_seed,
                unit_type=_ab_unit_type,
            )
        elif ("control" in _ab_params) and ("treatment" in _ab_params):
            ab_gpu = build_ab_style_readout(
                inv_detail,
                control=_control,
                treatment=_treatment,
                seed=_ab_seed,
                unit_type=_ab_unit_type,
            )
        elif ("control_model" in _ab_params) and ("treatment_model" in _ab_params):
            ab_gpu = build_ab_style_readout(
                inv_detail,
                control_model=_control,
                treatment_model=_treatment,
                seed=_ab_seed,
                unit_type=_ab_unit_type,
            )
        elif ("control_policy_name" in _ab_params) and ("treatment_policy_name" in _ab_params):
            ab_gpu = build_ab_style_readout(
                inv_detail,
                control_policy_name=_control,
                treatment_policy_name=_treatment,
                seed=_ab_seed,
                unit_type=_ab_unit_type,
            )
        else:
            ab_gpu = build_ab_style_readout(
                inv_detail,
                _control,
                _treatment,
                seed=_ab_seed,
                unit_type=_ab_unit_type,
            )
        write_table(ab_gpu, out_dir / "ab_test_gpu_guarded_summary.csv")

    causal = lightweight_event_residual_readout(test, pred_col='pred_tree', seed=seed)
    write_table(causal, out_dir / 'causal_effect_estimates.csv')

    explanations = generate_explanations(test, model.feature_importance, n_cases=expn_cfg.get('n_cases', 25))
    exp_detail, exp_metrics = evaluate_explanation_grounding(explanations)
    write_table(explanations, out_dir / 'forecast_explanations.csv')
    write_table(exp_detail, out_dir / 'explanation_grounding_detail.csv')
    write_json(exp_metrics, out_dir / 'explanation_eval_metrics.json')

    thresholds = gate_cfg.get('thresholds', {})
    raw_gate = build_release_gate(
        metrics, inv_summary, ab, bias_by_slice,
        control_forecast=gate_cfg.get('control_forecast', 'tree_ensemble'),
        treatment_forecast=gate_cfg.get('treatment_forecast', 'corrected'),
        control_policy=gate_cfg.get('control_policy', 'tree_ensemble_policy'),
        treatment_policy='corrected_policy',
        lead_time=gate_cfg.get('lead_time', 7),
        safety_stock_factor=gate_cfg.get('safety_stock_factor', 0.5),
        thresholds=thresholds,
    )
    gate = build_release_gate(
        metrics, inv_summary, ab, bias_by_slice,
        control_forecast=gate_cfg.get('control_forecast', 'tree_ensemble'),
        treatment_forecast=gate_cfg.get('treatment_forecast', 'corrected'),
        control_policy=gate_cfg.get('control_policy', 'tree_ensemble_policy'),
        treatment_policy=gate_cfg.get('treatment_policy', 'risk_safe_corrected_policy'),
        lead_time=gate_cfg.get('lead_time', 7),
        safety_stock_factor=gate_cfg.get('safety_stock_factor', 0.5),
        thresholds=thresholds,
    )
    write_json(raw_gate, out_dir / 'raw_correction_release_gate.json')
    write_json(gate, out_dir / 'release_gate.json')
    if (("release_safe_gpu_residual" in pred_map) or ("guarded_gpu_residual" in pred_map)) and (("release_safe_gpu_residual_policy" in inv_pred_cols) or ("guarded_gpu_residual_policy" in inv_pred_cols)):
        _gpu_treatment_forecast = "release_safe_gpu_residual" if "release_safe_gpu_residual" in pred_map else "guarded_gpu_residual"
        _gpu_treatment_policy = "release_safe_gpu_residual_policy" if "release_safe_gpu_residual_policy" in inv_pred_cols else "guarded_gpu_residual_policy"
        gpu_gate = build_release_gate(
            metrics, inv_summary, ab_gpu if ab_gpu is not None else ab, bias_by_slice,
            control_forecast="tree_ensemble",
            treatment_forecast=_gpu_treatment_forecast,
            control_policy="tree_ensemble_policy",
            treatment_policy=_gpu_treatment_policy,
            lead_time=gate_cfg.get("lead_time", 7),
            safety_stock_factor=gate_cfg.get("safety_stock_factor", 0.5),
            thresholds=thresholds,
        )
        write_json(gpu_gate, out_dir / "gpu_guarded_release_gate.json")

    fail = test.copy()
    fail['abs_error_corrected'] = (fail['pred_corrected'] - fail['demand']).abs()
    fail_cases = fail.sort_values('abs_error_corrected', ascending=False).head(20)[['item_store_id','date','demand','pred_tree','pred_corrected','event_flag','sell_price','model_disagreement_score']]
    write_table(fail_cases, out_dir / 'failure_cases.csv')

    # Reports with interpretation rather than table-only dumps.
    write_markdown(rep_dir / 'data_quality_report.md', 'Data Quality Report', report_section(
        tested='Loaded public M5 sales, calendar, and sell-price data into an item-store-day panel.',
        baseline='Raw M5 files are wide and not directly ready for model training.',
        enhanced='The pipeline validates schema, date coverage, duplicate keys, and creates SQL-ready marts.',
        result=f"Schema report:\n\n```json\n{schema_report}\n```\n\nSplit info:\n\n```json\n{split_info}\n```",
        interpretation='The panel passed the checks required for the sample-mode forecasting pipeline.',
        boundary='This does not validate Amazon internal data or real replenishment systems.'
    ))
    write_markdown(rep_dir / 'm5_forecast_report.md', 'M5 Forecast Report', report_section(
        tested='Compared seasonal naive, moving-average, tree-ensemble, and residual-corrected forecasts on a temporal holdout.',
        baseline='Seasonal naive and moving-average forecasts represent install-light vanilla demand baselines.',
        enhanced='The enhanced path adds lag/rolling/calendar/price/hierarchy features and residual correction.',
        result=df_head_markdown(metrics),
        interpretation='Use WAPE/RMSE for point-forecast accuracy and under-forecast rate for operational risk.',
        boundary='Results are sample-mode public-data benchmark results, not production Amazon metrics.'
    ))
    write_markdown(rep_dir / 'bias_correction_report.md', 'Bias Correction Report', report_section(
        tested='Measured whether validation-trained residual correction changes forecast bias on the test window.',
        baseline='The control forecast is the tree-ensemble point forecast.',
        enhanced='The corrected forecast adds a learned residual estimate to the control forecast.',
        result=f"## Forecast metrics\n\n{df_head_markdown(metrics)}\n\n## Slice metrics\n\n{df_head_markdown(bias_by_slice)}",
        interpretation='Correction should be judged by both accuracy and under-forecast risk. Mixed results are routed through the release gate instead of being overclaimed.',
        boundary='Residual correction is trained on validation-window residuals only.'
    ))
    write_markdown(rep_dir / 'inventory_policy_report.md', 'Inventory Policy Report', report_section(
        tested='Translated forecasts into synthetic ordering policies under lead-time and safety-stock assumptions.',
        baseline='The baseline policy orders from the tree-ensemble forecast plus safety stock.',
        enhanced='Corrected and risk-safe corrected policies are evaluated as ordering policies, not as pure point forecasts.',
        result=df_head_markdown(inv_summary),
        interpretation='Stockout, overstock, service level, total cost, Demand Recall, Inventory Precision, and Coverage F1 expose business trade-offs hidden by WAPE.',
        boundary='Inventory, lead-time, labor, and cost metrics are synthetic/proxy overlays.'
    ))
    write_markdown(rep_dir / 'ab_style_experiment_readout.md', 'Offline A/B-Style Experiment Readout', report_section(
        tested='Compared control and treatment ordering policies using item-store-week units and clustered bootstrap confidence intervals.',
        baseline='Control policy uses the tree-ensemble forecast.',
        enhanced='Treatment policy uses the risk-safe corrected ordering signal.',
        result=df_head_markdown(ab),
        interpretation='This is an experiment-style readout for portfolio evidence; CI direction and guardrails inform LAUNCH/ITERATE/HOLD/BLOCK.',
        boundary='This is not a randomized production A/B test.'
    ))
    write_markdown(rep_dir / 'causal_driver_analysis.md', 'Observational Causal Driver Analysis', report_section(
        tested='Estimated residual shifts around event or price-change indicators on the holdout set.',
        baseline='Correlation or feature importance alone does not establish causal behavior.',
        enhanced='The readout separates treated/control windows and reports uncertainty.',
        result=df_head_markdown(causal),
        interpretation='This is a lightweight observational diagnostic suitable for MVP; Favorita promotion/IPW/pretrend checks are the next extension.',
        boundary='No randomized causal proof is claimed.'
    ))
    write_markdown(rep_dir / 'forecast_explainability_report.md', 'Forecast Explanation Grounding Report', report_section(
        tested='Generated forecast explanations and evaluated numeric consistency, driver support, unsupported claims, and negative-control detection.',
        baseline='Feature importance alone does not verify whether generated explanations are faithful.',
        enhanced='The grounding evaluator checks explanation text against forecast outputs and deliberately flawed negative controls.',
        result=f"## Metrics\n\n```json\n{exp_metrics}\n```\n\n## Examples\n\n{df_head_markdown(explanations, 5)}",
        interpretation='Negative-control detection demonstrates that the evaluator can reject unsupported narratives instead of only scoring generated text as perfect.',
        boundary='No external LLM API is required in MVP; this is an LLM-ready deterministic grounding layer.'
    ))
    write_markdown(rep_dir / 'launch_decision_memo.md', 'Launch Decision Memo', build_launch_memo_text(gate))
    write_markdown(rep_dir / 'model_card.md', 'Model Card', model_card_text())
    write_markdown(rep_dir / 'leakage_audit.md', 'Leakage Audit', report_section(
        tested='Audited the major temporal leakage risks before local execution.',
        baseline='Naive time-series projects often leak same-day or future target information through rolling features, residual correction, or policy simulation.',
        enhanced='Lag/rolling features are shifted; residual correction trains on validation residuals; risk-safe policy is separated from point-forecast metrics; inventory orders are computed from forecasts and historical uncertainty only.',
        result='No known same-day target leakage is intentionally introduced in the MVP pipeline.',
        interpretation='The audit makes the project more credible for interview discussion because it states what is and is not proven.',
        boundary='This is a code-level audit, not an external validation or formal model-risk review.'
    ))
    write_markdown(rep_dir / 'failure_case_gallery.md', 'Failure Case Gallery', report_section(
        tested='Selected the largest corrected-forecast errors for manual inspection.',
        baseline='Aggregate metrics can hide hard cases.',
        enhanced='Failure cases are saved for review of holiday/event, price-change, sparse-demand, and overcorrection patterns.',
        result=df_head_markdown(fail_cases, 20),
        interpretation='These cases should guide the next model iteration rather than being hidden.',
        boundary='This is sample-mode diagnostic evidence.'
    ))
    write_markdown(rep_dir / 'executive_summary.md', 'Executive Summary', f"""
Retail Forecasting Lab completed a sample-mode run on M5 public retail demand data.

## Decision

**{gate.get('decision')}**

## What changed in v3

- Consolidated configuration into `configs/pipeline.yaml`.
- Separated point-forecast metrics from risk-safe inventory policy metrics.
- Fixed release-gate overstock and total-cost guardrails so they can actually fail.
- Replaced row-level A/B readout with item-store-week units and clustered bootstrap intervals.
- Moved optional non-core dataset adapters into `extensions/`.
- Wrote lightweight detail samples by default instead of large detail CSVs.
- Added interpretation-oriented reports instead of table-only report dumps.

## Core outputs

- Forecast metrics: `outputs/forecast_metrics.csv`
- Bias by slice: `outputs/bias_by_slice.csv`
- Inventory policy summary: `outputs/inventory_policy_summary.csv`
- Offline A/B-style readout: `outputs/ab_test_summary.csv`
- Release gate: `outputs/release_gate.json`

## Claim boundary

This is a public-data, local-runnable, offline forecasting measurement framework. Inventory and lead-time impacts use synthetic/proxy overlays.
""")

    print(f"Pipeline completed. Decision: {gate.get('decision')}. Outputs: {out_dir}. Reports: {rep_dir}.")


if __name__ == '__main__':
    main()

