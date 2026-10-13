"""Data pipeline: `from src.data import build_dataloaders` (see dataset.py).

Names from dataset.py are loaded lazily so `python -m src.data.splits` does not import torch/cv2.
"""

import importlib

_DATASET_NAMES = {
    "SAMPLING_STRATEGIES", "UCF101Subset", "VideoTransform", "build_dataloaders", "count_frames",
    "get_preset", "load_clip", "sample_indices", "seed_everything", "seed_worker",
}


def __getattr__(name):
    if name in _DATASET_NAMES:
        return getattr(importlib.import_module("src.data.dataset"), name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
