"""Smoke test for the loader on tiny synthetic .avi files (test-only data, never used for results)."""

import csv
import pickle

import cv2
import numpy as np
import pytest
import torch

from src.data import build_dataloaders, get_preset, load_clip, sample_indices
from src.data.splits import CLASSES

NUM_FRAMES = 8


def write_avi(path, num_frames, h=120, w=160, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (w, h))
    assert writer.isOpened(), "OpenCV cannot write MJPG .avi in this environment"
    rng = np.random.default_rng(seed)
    for _ in range(num_frames):
        writer.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture()
def fake_dataset(tmp_path):
    root = tmp_path / "UCF-101"
    rows = []
    for ci, c in enumerate(CLASSES):
        for split, group in (("train", 8), ("val", 9), ("test", 1)):
            rel = f"{c}/v_{c}_g{group:02d}_c01.avi"
            # one class gets clips shorter than NUM_FRAMES to exercise frame repetition
            n = 5 if c == "Lunges" else 20
            write_avi(root / rel, n, seed=ci)
            rows.append({"path": rel, "label": c, "group": f"g{group:02d}", "split": split})
    csv_path = tmp_path / "split.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "split"])
        w.writeheader()
        w.writerows(rows)
    return root, csv_path


@pytest.mark.parametrize("sampling", ["uniform", "contiguous"])
def test_loader_shapes(fake_dataset, sampling):
    root, csv_path = fake_dataset
    train, val, test, names = build_dataloaders(root, csv_path, sampling=sampling, num_frames=NUM_FRAMES,
                                                batch_size=4, num_workers=0, seed=42)
    assert names == sorted(CLASSES)
    crop = get_preset()["crop_size"]
    for loader in (train, val, test):
        videos, labels = next(iter(loader))
        assert videos.shape == (4, 3, NUM_FRAMES, crop[0], crop[1])
        assert videos.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert 0 <= labels.min() and labels.max() < len(CLASSES)
        assert torch.isfinite(videos).all()


def test_dataset_is_picklable(fake_dataset):
    """Workers are started with spawn/forkserver on Windows and Python 3.14+, which pickles the dataset."""
    root, csv_path = fake_dataset
    train = build_dataloaders(root, csv_path, num_frames=NUM_FRAMES, num_workers=0)[0]
    pickle.loads(pickle.dumps(train.dataset))


def test_eval_loader_is_deterministic(fake_dataset):
    root, csv_path = fake_dataset
    a = next(iter(build_dataloaders(root, csv_path, sampling="contiguous", num_frames=NUM_FRAMES,
                                    batch_size=4, num_workers=0)[2]))[0]
    b = next(iter(build_dataloaders(root, csv_path, sampling="contiguous", num_frames=NUM_FRAMES,
                                    batch_size=4, num_workers=0)[2]))[0]
    assert torch.equal(a, b)


def test_short_clip_repeats_last_frame(fake_dataset):
    root, _ = fake_dataset
    clip = load_clip(root / "Lunges/v_Lunges_g08_c01.avi", NUM_FRAMES, "uniform")
    assert clip.shape[0] == NUM_FRAMES
    assert torch.equal(clip[4], clip[-1])


def test_sample_indices():
    assert sample_indices(100, 4, "uniform", train=False) == [0, 33, 66, 99]
    assert sample_indices(100, 10, "contiguous", train=False) == list(range(45, 55))
    assert sample_indices(3, 5, "uniform", train=True) == [0, 1, 2, 2, 2]
    idx = sample_indices(100, 10, "contiguous", train=True)
    assert idx == list(range(idx[0], idx[0] + 10))


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="bad.avi"):
        load_clip(bad, NUM_FRAMES)
