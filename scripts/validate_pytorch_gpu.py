import json
import time
from pathlib import Path

import torch

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

result = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda_version": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(4096, 4096, device=device)
if device == "cuda":
    torch.cuda.synchronize()

start = time.time()
y = x @ x

if device == "cuda":
    torch.cuda.synchronize()

result["smoke_test_device"] = device
result["matmul_elapsed_sec"] = round(time.time() - start, 6)
result["matmul_mean"] = float(y.mean().item())

with open(out_dir / "gpu_pytorch_validation.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
