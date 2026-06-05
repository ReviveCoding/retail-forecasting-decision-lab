from pathlib import Path

path = Path("scripts/run_guarded_gpu_residual_postprocess.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    'raw["mean_bias"].abs()',
    'abs(raw["mean_bias"])'
)

path.write_text(text, encoding="utf-8")
print("Patched scalar abs() issue in run_guarded_gpu_residual_postprocess.py")
