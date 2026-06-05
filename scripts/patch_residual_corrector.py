from pathlib import Path

path = Path("scripts/run_torch_residual_corrector_smoke.py")
text = path.read_text(encoding="utf-8")

old = '''    exclude = {
        "demand",
        "pred_corrected",
        "order_signal_risk_safe_corrected",
        "pred_p10",
        "pred_p90",
    }
'''

new = '''    exclude = {
        "demand",
        "pred_tree",
        "pred_corrected",
        "order_signal_risk_safe_corrected",
        "pred_p10",
        "pred_p90",
    }
'''

if old not in text:
    raise SystemExit("Target exclude block not found. Open the script and check the exclude block manually.")

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")

print("Patched: pred_tree is now excluded from numeric feature columns to prevent duplicate pred_tree labels.")
