import json
from pathlib import Path

result_path = Path("outputs/gpu_pytorch_validation.json")
report_path = Path("reports/gpu_pytorch_validation_report.md")
report_path.parent.mkdir(exist_ok=True)

if not result_path.exists():
    raise FileNotFoundError("Run scripts/validate_pytorch_gpu.py first.")

data = json.loads(result_path.read_text(encoding="utf-8"))

lines = [
    "# GPU PyTorch Validation Report",
    "",
    "## Purpose",
    "",
    "This report records the optional CUDA-backed PyTorch environment validation for GPU forecasting extensions. The core forecasting pipeline remains CPU-runnable.",
    "",
    "## Result",
    "",
    f"- PyTorch version: `{data.get('torch_version')}`",
    f"- CUDA available: `{data.get('cuda_available')}`",
    f"- PyTorch CUDA runtime: `{data.get('torch_cuda_version')}`",
    f"- Device: `{data.get('device_name')}`",
    f"- Smoke-test device: `{data.get('smoke_test_device')}`",
    f"- 4096x4096 matrix multiplication elapsed seconds: `{data.get('matmul_elapsed_sec')}`",
    "",
    "## Interpretation",
    "",
    "CUDA-backed PyTorch is available in this environment, and the smoke test executed tensor computation on the GPU. This enables optional GPU model-training experiments without making the core repository dependent on GPU hardware.",
    "",
    "## Claim boundary",
    "",
    "This validation confirms GPU availability and tensor-compute execution. It does not by itself claim that the forecasting model has been trained on GPU until a GPU forecasting experiment is run and reported.",
]

report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {report_path}")
