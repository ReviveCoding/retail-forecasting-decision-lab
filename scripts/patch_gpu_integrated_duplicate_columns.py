from pathlib import Path

path = Path("src/models/torch_guarded_residual_corrector.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    '    feature_cols = feature_cols + ["pred_gpu_base_tree"]',
    '    feature_cols = list(dict.fromkeys(feature_cols + ["pred_gpu_base_tree"]))'
)

text = text.replace(
    '    required = feature_cols + ["demand", "pred_gpu_base_tree"]',
    '    required = list(dict.fromkeys(feature_cols + ["demand"]))'
)

text = text.replace(
    '    residual_fit_data = residual_fit[required].replace([np.inf, -np.inf], np.nan).dropna()',
    '    residual_fit_data = residual_fit[required].replace([np.inf, -np.inf], np.nan).dropna()' + "\n" +
    '    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()' + "\n" +
    '    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()' + "\n" +
    '    for name, data_frame in [("residual_fit_data", residual_fit_data), ("guard_data", guard_data), ("test_data", test_data)]:' + "\n" +
    '        if data_frame.columns.duplicated().any():' + "\n" +
    '            dupes = data_frame.columns[data_frame.columns.duplicated()].tolist()' + "\n" +
    '            raise ValueError(f"Duplicate columns in {name}: {dupes}")'
)

# Remove duplicated guard/test assignment block if the replacement created duplicates.
dup = '    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()\n    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()\n\n    if len(residual_fit_data) < 100'
text = text.replace(
    '    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()\n    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()\n    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()\n    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()\n\n    if len(residual_fit_data) < 100',
    '    guard_data = guard_val[required].replace([np.inf, -np.inf], np.nan).dropna()\n    test_data = test_out[required].replace([np.inf, -np.inf], np.nan).dropna()\n\n    if len(residual_fit_data) < 100'
)

path.write_text(text, encoding="utf-8")
print("Patched duplicate pred_gpu_base_tree handling in torch_guarded_residual_corrector.py")
