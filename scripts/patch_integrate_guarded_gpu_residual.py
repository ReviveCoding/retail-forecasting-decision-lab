from pathlib import Path

path = Path("src/run_pipeline.py")
text = path.read_text(encoding="utf-8")

if "from src.models.torch_guarded_residual_corrector import apply_guarded_gpu_residual_stage" not in text:
    text = text.replace(
        "from src.models.forecast_uncertainty import add_candidate_forecast_uncertainty\n",
        "from src.models.forecast_uncertainty import add_candidate_forecast_uncertainty\nfrom src.models.torch_guarded_residual_corrector import apply_guarded_gpu_residual_stage\n"
    )

if "gpu_cfg = cfg.get(" not in text:
    text = text.replace(
        "    expn_cfg = cfg.get(" + chr(39) + "explainability" + chr(39) + ", {})\n",
        "    expn_cfg = cfg.get(" + chr(39) + "explainability" + chr(39) + ", {})\n    gpu_cfg = cfg.get(" + chr(39) + "gpu_residual" + chr(39) + ", {})\n"
    )

lines = text.splitlines()
out = []
inserted_stage = "apply_guarded_gpu_residual_stage(" in text
inserted_pred = "pred_map[" + chr(34) + "guarded_gpu_residual" + chr(34) + "]" in text
inserted_inv = "guarded_gpu_residual_policy" in text
inserted_ab_gpu = "ab_test_gpu_guarded_summary.csv" in text
inserted_gpu_gate = "gpu_guarded_release_gate.json" in text

in_pred_map = False
in_inv_map = False

for line in lines:
    out.append(line)

    if (not inserted_stage) and "test = add_candidate_forecast_uncertainty" in line:
        out.extend([
            "    if bool(gpu_cfg.get(" + chr(34) + "enabled" + chr(34) + ", False)):",
            "        gpu_stage = apply_guarded_gpu_residual_stage(",
            "            train=train,",
            "            val=val,",
            "            test=test,",
            "            model_cfg=model_cfg,",
            "            gpu_cfg=gpu_cfg,",
            "            seed=seed,",
            "            out_dir=out_dir,",
            "            rep_dir=rep_dir,",
            "        )",
            "        test = gpu_stage.get(" + chr(34) + "test" + chr(34) + ", test)",
        ])
        inserted_stage = True

    if "pred_map = {" in line:
        in_pred_map = True
    elif in_pred_map and line.strip() == "}":
        in_pred_map = False
        if not inserted_pred:
            out.extend([
                "    if " + chr(34) + "pred_guarded_gpu_residual" + chr(34) + " in test.columns:",
                "        pred_map[" + chr(34) + "guarded_gpu_residual" + chr(34) + "] = " + chr(34) + "pred_guarded_gpu_residual" + chr(34),
            ])
            inserted_pred = True

    if "inv_pred_cols = {" in line:
        in_inv_map = True
    elif in_inv_map and line.strip() == "}":
        in_inv_map = False
        if not inserted_inv:
            out.extend([
                "    if " + chr(34) + "order_signal_guarded_gpu_residual" + chr(34) + " in test.columns:",
                "        inv_pred_cols[" + chr(34) + "guarded_gpu_residual_policy" + chr(34) + "] = " + chr(34) + "order_signal_guarded_gpu_residual" + chr(34),
            ])
            inserted_inv = True

    if (not inserted_ab_gpu) and "ab_test_summary.csv" in line:
        out.extend([
            "    ab_gpu = None",
            "    if " + chr(34) + "guarded_gpu_residual_policy" + chr(34) + " in inv_pred_cols:",
            "        ab_gpu = build_ab_style_readout(",
            "            inv_detail,",
            "            control_policy=" + chr(34) + "tree_ensemble_policy" + chr(34) + ",",
            "            treatment_policy=" + chr(34) + "guarded_gpu_residual_policy" + chr(34) + ",",
            "            seed=exp_cfg.get(" + chr(34) + "random_seed" + chr(34) + ", seed),",
            "            unit_type=exp_cfg.get(" + chr(34) + "unit_type" + chr(34) + ", " + chr(34) + "item-store-week" + chr(34) + "),",
            "        )",
            "        write_table(ab_gpu, out_dir / " + chr(34) + "ab_test_gpu_guarded_summary.csv" + chr(34) + ")",
        ])
        inserted_ab_gpu = True

    if (not inserted_gpu_gate) and "write_json(gate, out_dir / " in line and "release_gate.json" in line:
        out.extend([
            "    if (" + chr(34) + "guarded_gpu_residual" + chr(34) + " in pred_map) and (" + chr(34) + "guarded_gpu_residual_policy" + chr(34) + " in inv_pred_cols):",
            "        gpu_gate = build_release_gate(",
            "            metrics, inv_summary, ab_gpu if ab_gpu is not None else ab, bias_by_slice,",
            "            control_forecast=" + chr(34) + "tree_ensemble" + chr(34) + ",",
            "            treatment_forecast=" + chr(34) + "guarded_gpu_residual" + chr(34) + ",",
            "            control_policy=" + chr(34) + "tree_ensemble_policy" + chr(34) + ",",
            "            treatment_policy=" + chr(34) + "guarded_gpu_residual_policy" + chr(34) + ",",
            "            lead_time=gate_cfg.get(" + chr(34) + "lead_time" + chr(34) + ", 7),",
            "            safety_stock_factor=gate_cfg.get(" + chr(34) + "safety_stock_factor" + chr(34) + ", 0.5),",
            "            thresholds=thresholds,",
            "        )",
            "        write_json(gpu_gate, out_dir / " + chr(34) + "gpu_guarded_release_gate.json" + chr(34) + ")",
        ])
        inserted_gpu_gate = True

path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("Patched src/run_pipeline.py for leakage-safe guarded GPU residual integration.")
