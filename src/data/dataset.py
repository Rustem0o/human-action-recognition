"""Video loading, frame sampling, transforms and DataLoaders for the UCF101 subset.

Main entry point for teammates:

    from src.data import build_dataloaders
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        video_root="data/UCF-101", split_csv="splits/split01_seed42.csv")

Each batch is (videos, labels): videos float32 (B, C, T, H, W) normalized, labels int64 (B,).
"""

import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R2Plus1D_18_Weights

from src.data.splits import class_names as default_class_names
from src.data.splits import read_csv

SAMPLING_STRATEGIES = ("uniform", "contiguous")


# --------------------------------------------------------------------------- reproducibility

def seed_everything(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker_init_fn: derive numpy/random seeds from the torch seed of this worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- frame sampling

def sample_indices(num_total, num_frames, strategy, train, rng=random):
    """Return `num_frames` frame indices into a clip with `num_total` frames.

    uniform:    evenly spaced over the whole clip.
    contiguous: one window of consecutive frames; random start if train, centered otherwise.
    If the clip is shorter than `num_frames`, all frames are used and the last one is repeated.
    """
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unknown sampling strategy '{strategy}', expected one of {SAMPLING_STRATEGIES}")
    if num_total <= 0:
        raise ValueError("Clip has no frames")
    if num_total < num_frames:
        return list(range(num_total)) + [num_total - 1] * (num_frames - num_total)
    if strategy == "uniform":
        return np.linspace(0, num_total - 1, num_frames).round().astype(int).tolist()
    max_start = num_total - num_frames
    start = rng.randint(0, max_start) if train else max_start // 2
    return list(range(start, start + num_frames))


def _decode(path, wanted):
    """Decode frames with indices in `wanted` (sorted set) sequentially. Returns (dict idx->RGB frame, n_read)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path} (missing, corrupt or unsupported codec)")
    frames = {}
    last = max(wanted) if wanted else -1
    idx = 0
    try:
        while idx <= last:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to decode frame {idx} of {path}")
                frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
    return frames, idx


def count_frames(path):
    """Exact frame count by grabbing every frame (CAP_PROP_FRAME_COUNT can be wrong for AVI)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path} (missing, corrupt or unsupported codec)")
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def load_clip(path, num_frames=16, strategy="uniform", train=False, rng=random):
    """Load `num_frames` RGB frames as a uint8 tensor (T, C, H, W)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path} (missing, corrupt or unsupported codec)")
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    num_total = reported if reported > 0 else count_frames(path)
    if num_total <= 0:
        raise RuntimeError(f"Video has no decodable frames: {path}")
    indices = sample_indices(num_total, num_frames, strategy, train, rng)
    frames, n_read = _decode(path, set(indices))

    if len(frames) < len(set(indices)):
        # The header over-reported the frame count: resample with the real count.
        num_total = n_read if n_read > 0 else count_frames(path)
        if num_total <= 0:
            raise RuntimeError(f"Video has no decodable frames: {path}")
        indices = sample_indices(num_total, num_frames, strategy, train, rng)
        frames, _ = _decode(path, set(indices))
        if len(frames) < len(set(indices)):
            raise RuntimeError(f"Could not decode the requested frames of {path}")

    clip = np.stack([frames[i] for i in indices])  # (T, H, W, C) uint8
    return torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)


# --------------------------------------------------------------------------- transforms

def get_preset(weights=None):
    """Resize/crop/mean/std from the torchvision weights preset (default: R(2+1)D-18 Kinetics-400)."""
    weights = weights or R2Plus1D_18_Weights.KINETICS400_V1
    t = weights.transforms()
    return {
        "resize_size": list(t.resize_size),
        "crop_size": list(t.crop_size),
        "mean": list(t.mean),
        "std": list(t.std),
    }


class VideoTransform:
    """uint8 (T, C, H, W) -> float normalized (C, T, H, W).

    train: resize -> random crop -> random horizontal flip -> normalize
    eval:  resize -> center crop -> normalize
    The same crop/flip is applied to all frames of a clip.
    """

    def __init__(self, preset, train):
        self.resize_size = preset["resize_size"]
        self.crop_h, self.crop_w = preset["crop_size"]
        self.mean = torch.tensor(preset["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(preset["std"]).view(1, 3, 1, 1)
        self.train = train

    def __call__(self, clip):
        x = clip.float() / 255.0
        x = TF.resize(x, self.resize_size, antialias=True)
        h, w = x.shape[-2:]
        if h < self.crop_h or w < self.crop_w:
            raise ValueError(f"Resized clip {h}x{w} smaller than crop {self.crop_h}x{self.crop_w}")
        if self.train:
            # module-level `random` is seeded per worker by seed_worker()
            top = random.randint(0, h - self.crop_h)
            left = random.randint(0, w - self.crop_w)
            x = x[..., top:top + self.crop_h, left:left + self.crop_w]
            if random.random() < 0.5:
                x = x.flip(-1)
        else:
            x = TF.center_crop(x, [self.crop_h, self.crop_w])
        x = (x - self.mean) / self.std
        return x.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)


# --------------------------------------------------------------------------- dataset

class UCF101Subset(Dataset):
    """Clips of one split from the split CSV. `samples` is a list of (relative_path, label_idx)."""

    def __init__(self, video_root, split_csv, split, class_names=None, sampling="uniform",
                 num_frames=16, weights=None):
        if sampling not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown sampling strategy '{sampling}', expected one of {SAMPLING_STRATEGIES}")
        self.video_root = str(video_root)
        self.split = split
        self.sampling = sampling
        self.num_frames = num_frames
        self.train = split == "train"
        self.class_names = list(class_names) if class_names is not None else default_class_names()
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        rows = [r for r in read_csv(split_csv) if r["split"] == split]
        if not rows:
            raise ValueError(f"No rows with split='{split}' in {split_csv}")
        unknown = sorted({r["label"] for r in rows} - set(self.class_to_idx))
        if unknown:
            raise ValueError(f"Labels in {split_csv} not in class_names: {unknown}")
        self.samples = [(r["path"], self.class_to_idx[r["label"]]) for r in rows]
        self.transform = VideoTransform(get_preset(weights), train=self.train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        path = os.path.join(self.video_root, rel_path)
        try:
            clip = load_clip(path, self.num_frames, self.sampling, train=self.train)
        except Exception as e:
            raise RuntimeError(f"Failed to load sample {idx} ({rel_path}) of split '{self.split}': {e}") from e
        return self.transform(clip), label


# --------------------------------------------------------------------------- loaders

def build_dataloaders(video_root, split_csv, sampling="uniform", num_frames=16, batch_size=8,
                      num_workers=2, seed=42, weights=None):
    """Return train_loader, val_loader, test_loader, class_names.

    `weights` (optional) selects the torchvision preset for resize/crop/mean/std;
    default is R2Plus1D_18_Weights.KINETICS400_V1. Pass e.g. Swin3D_T_Weights.KINETICS400_V1
    to get matching preprocessing for Video Swin-T.
    Only train_loader is shuffled; val/test keep CSV order, so
    test_loader.dataset.samples[i] is the (path, label) of the i-th test prediction.
    """
    seed_everything(seed)
    names = default_class_names()
    datasets = {
        s: UCF101Subset(video_root, split_csv, s, names, sampling, num_frames, weights)
        for s in ("train", "val", "test")
    }

    def make_loader(split, shuffle):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=num_workers > 0,
        )

    return make_loader("train", True), make_loader("val", False), make_loader("test", False), names
