from pathlib import Path

path = Path("src/models/torch_guarded_residual_corrector.py")
text = path.read_text(encoding="utf-8")

old = '    test_out["order_signal_guarded_gpu_residual"] = test_out["pred_guarded_gpu_residual"]'
new = ''    # Forecast metric and ordering-policy signal are intentionally separated.
    # The guarded point forecast preserves WAPE improvement, while the release-safe
    # signal floors the GPU forecast at the original tree forecast to reduce
    # under-forecast and stockout risk without using the test window for tuning.
    if "pred_tree" in test_out.columns:
        test_out["pred_release_safe_gpu_residual"] = test_out[["pred_guarded_gpu_residual", "pred_tree"]].max(axis=1)
    else:
        test_out["pred_release_safe_gpu_residual"] = test_out[["pred_guarded_gpu_residual", "pred_gpu_base_tree"]].max(axis=1)

    test_out["order_signal_guarded_gpu_residual"] = test_out["pred_guarded_gpu_residual"]
    test_out["order_signal_release_safe_gpu_residual"] = test_out["pred_release_safe_gpu_residual"]''

if old not in text:
    raise SystemExit("Target order_signal line not found. Check torch_guarded_residual_corrector.py manually.")

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print("Patched module with release-safe GPU residual forecast and policy signal.")
