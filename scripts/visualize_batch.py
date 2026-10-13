"""Save a figure with frames from one batch and print the average loading time per batch.

Usage:
    python scripts/visualize_batch.py --video-root data/UCF-101 --split-csv splits/split01_seed42.csv \
        --split train --sampling uniform --out results/figures/batch_train_uniform.png
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import build_dataloaders, get_preset  # noqa: E402


def denormalize(video, mean, std):
    """(C, T, H, W) normalized -> (T, H, W, C) in [0, 1]."""
    mean = torch.tensor(mean).view(3, 1, 1, 1)
    std = torch.tensor(std).view(3, 1, 1, 1)
    return (video * std + mean).clamp(0, 1).permute(1, 2, 3, 0).numpy()


def save_figure(videos, labels, class_names, preset, out_path, max_clips=4, max_frames=8):
    n_clips = min(max_clips, videos.shape[0])
    t = videos.shape[2]
    frame_ids = np.linspace(0, t - 1, min(max_frames, t)).round().astype(int)
    fig, axes = plt.subplots(n_clips, len(frame_ids), figsize=(1.6 * len(frame_ids), 1.7 * n_clips), squeeze=False)
    for i in range(n_clips):
        frames = denormalize(videos[i], preset["mean"], preset["std"])
        for j, f in enumerate(frame_ids):
            ax = axes[i, j]
            ax.imshow(frames[f])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(class_names[int(labels[i])], fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-root", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--sampling", choices=["uniform", "contiguous"], default="uniform")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-batches", type=int, default=10, help="batches to time (after the first one)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/figures/batch.png")
    args = p.parse_args(argv)

    loaders = build_dataloaders(args.video_root, args.split_csv, args.sampling, args.num_frames,
                                args.batch_size, args.num_workers, args.seed)
    loader = dict(zip(["train", "val", "test"], loaders[:3]))[args.split]
    class_names = loaders[3]

    times = []
    first_batch = None
    t0 = time.perf_counter()
    for i, (videos, labels) in enumerate(loader):
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if first_batch is None:
            first_batch = (videos, labels)
            print(f"batch shape: {tuple(videos.shape)} dtype={videos.dtype}, labels: {labels.tolist()}")
        if i >= args.num_batches:
            break
        t0 = time.perf_counter()

    save_figure(*first_batch, class_names, get_preset(), args.out)
    print(f"Saved figure to {args.out}")
    print(f"first batch (includes worker start-up): {times[0]:.3f} s")
    if len(times) > 1:
        print(f"average over next {len(times) - 1} batches: {np.mean(times[1:]):.3f} s/batch "
              f"(batch_size={args.batch_size}, num_workers={args.num_workers}, sampling={args.sampling})")


if __name__ == "__main__":
    main()
