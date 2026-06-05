from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)

    old_crlf = old.replace("\n", "\r\n")
    new_crlf = new.replace("\n", "\r\n")
    if old_crlf in text:
        return text.replace(old_crlf, new_crlf, 1)

    raise SystemExit(f"Patch target not found: {label}")


# 1) Patch GPU residual module: separate point forecast and release-safe ordering signal.
module_path = Path("src/models/torch_guarded_residual_corrector.py")
module_text = module_path.read_text(encoding="utf-8")

old_module = '    test_out["order_signal_guarded_gpu_residual"] = test_out["pred_guarded_gpu_residual"]'
new_module = '''    # Forecast metric and ordering-policy signal are intentionally separated.
    # The guarded point forecast is used for WAPE evaluation.
    # The release-safe signal floors the GPU forecast at the tree forecast to reduce
    # under-forecast and stockout risk without using the test window for tuning.
    if "pred_tree" in test_out.columns:
        test_out["pred_release_safe_gpu_residual"] = test_out[["pred_guarded_gpu_residual", "pred_tree"]].max(axis=1)
    else:
        test_out["pred_release_safe_gpu_residual"] = test_out[["pred_guarded_gpu_residual", "pred_gpu_base_tree"]].max(axis=1)

    test_out["order_signal_guarded_gpu_residual"] = test_out["pred_guarded_gpu_residual"]
    test_out["order_signal_release_safe_gpu_residual"] = test_out["pred_release_safe_gpu_residual"]'''

if "pred_release_safe_gpu_residual" not in module_text:
    module_text = replace_once(module_text, old_module, new_module, "module release-safe signal")
    module_path.write_text(module_text, encoding="utf-8")
    print("Patched module release-safe GPU signal.")
else:
    print("Module release-safe GPU signal already present.")


# 2) Patch run_pipeline.py: evaluate release-safe forecast and policy.
pipeline_path = Path("src/run_pipeline.py")
text = pipeline_path.read_text(encoding="utf-8")

old_pred = '''    if "pred_guarded_gpu_residual" in test.columns:
        pred_map["guarded_gpu_residual"] = "pred_guarded_gpu_residual"
'''
new_pred = '''    if "pred_guarded_gpu_residual" in test.columns:
        pred_map["guarded_gpu_residual"] = "pred_guarded_gpu_residual"
    if "pred_release_safe_gpu_residual" in test.columns:
        pred_map["release_safe_gpu_residual"] = "pred_release_safe_gpu_residual"
'''

if 'pred_map["release_safe_gpu_residual"]' not in text:
    text = replace_once(text, old_pred, new_pred, "pred_map release_safe_gpu_residual")
    print("Patched pred_map release-safe forecast.")
else:
    print("pred_map release-safe forecast already present.")


old_inv = '''    if "order_signal_guarded_gpu_residual" in test.columns:
        inv_pred_cols["guarded_gpu_residual_policy"] = "order_signal_guarded_gpu_residual"
'''
new_inv = '''    if "order_signal_guarded_gpu_residual" in test.columns:
        inv_pred_cols["guarded_gpu_residual_policy"] = "order_signal_guarded_gpu_residual"
    if "order_signal_release_safe_gpu_residual" in test.columns:
        inv_pred_cols["release_safe_gpu_residual_policy"] = "order_signal_release_safe_gpu_residual"
'''

if 'inv_pred_cols["release_safe_gpu_residual_policy"]' not in text:
    text = replace_once(text, old_inv, new_inv, "inventory release_safe_gpu_residual_policy")
    print("Patched inventory release-safe policy.")
else:
    print("inventory release-safe policy already present.")


old_ab_if = '    if "guarded_gpu_residual_policy" in inv_pred_cols:\n'
new_ab_if = '    if ("release_safe_gpu_residual_policy" in inv_pred_cols) or ("guarded_gpu_residual_policy" in inv_pred_cols):\n'

if new_ab_if not in text:
    text = replace_once(text, old_ab_if, new_ab_if, "GPU A/B policy availability")
    print("Patched GPU A/B policy condition.")
else:
    print("GPU A/B policy condition already present.")


old_treatment = '        _treatment = "guarded_gpu_residual_policy"\n'
new_treatment = '        _treatment = "release_safe_gpu_residual_policy" if "release_safe_gpu_residual_policy" in inv_pred_cols else "guarded_gpu_residual_policy"\n'

if new_treatment not in text:
    text = replace_once(text, old_treatment, new_treatment, "GPU A/B treatment policy")
    print("Patched GPU A/B treatment policy.")
else:
    print("GPU A/B treatment policy already present.")


old_gate = '''    if ("guarded_gpu_residual" in pred_map) and ("guarded_gpu_residual_policy" in inv_pred_cols):
        gpu_gate = build_release_gate(
'''
new_gate = '''    if (("release_safe_gpu_residual" in pred_map) or ("guarded_gpu_residual" in pred_map)) and (("release_safe_gpu_residual_policy" in inv_pred_cols) or ("guarded_gpu_residual_policy" in inv_pred_cols)):
        _gpu_treatment_forecast = "release_safe_gpu_residual" if "release_safe_gpu_residual" in pred_map else "guarded_gpu_residual"
        _gpu_treatment_policy = "release_safe_gpu_residual_policy" if "release_safe_gpu_residual_policy" in inv_pred_cols else "guarded_gpu_residual_policy"
        gpu_gate = build_release_gate(
'''

if "_gpu_treatment_forecast" not in text:
    text = replace_once(text, old_gate, new_gate, "GPU release gate treatment selector")
    print("Patched GPU release gate selector.")
else:
    print("GPU release gate selector already present.")


text = text.replace(
    '            treatment_forecast="guarded_gpu_residual",',
    '            treatment_forecast=_gpu_treatment_forecast,'
)
text = text.replace(
    '            treatment_policy="guarded_gpu_residual_policy",',
    '            treatment_policy=_gpu_treatment_policy,'
)

pipeline_path.write_text(text, encoding="utf-8")
print("Done: release-safe GPU residual pipeline patch applied.")
