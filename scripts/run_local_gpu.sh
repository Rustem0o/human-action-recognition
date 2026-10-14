#!/usr/bin/env bash
# Run from the normal Linux terminal so NVIDIA devices and network are visible.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p results/logs
exec > >(tee -a results/logs/baseline_local_gpu.log) 2>&1

nvidia-smi
if command -v uv >/dev/null 2>&1; then
    if [[ ! -x .venv-gpu/bin/python ]]; then uv venv --python python3 .venv-gpu; fi
    uv pip install --python .venv-gpu/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-gpu/bin/python 'numpy>=1.24' 'opencv-python-headless>=4.8' 'pytest>=7'
else
    if [[ ! -x .venv-gpu/bin/python ]]; then python3 -m venv .venv-gpu; fi
    .venv-gpu/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    .venv-gpu/bin/python -m pip install 'numpy>=1.24' 'opencv-python-headless>=4.8' 'pytest>=7'
fi

.venv-gpu/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable in this terminal. Stop before downloading the dataset.')
x = torch.ones(1, device='cuda')
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('GPU tensor check:', x.sum().item())
PY
OMP_NUM_THREADS=4 .venv-gpu/bin/python -m pytest -q
.venv-gpu/bin/python scripts/prepare_ucf101.py
.venv-gpu/bin/python -u scripts/train_baseline.py --device cuda --epochs 3 --batch-size 4 --num-workers 2 "$@"
