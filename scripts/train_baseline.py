"""Train R(2+1)D-18 and select a checkpoint using validation only.

Example: python scripts/train_baseline.py --device cuda --epochs 3 --batch-size 4
The official test set is evaluated only when --evaluate-test is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dataloaders  # noqa: E402
from src.data.splits import assert_no_group_overlap, check_files_exist, read_csv  # noqa: E402
from src.evaluation import confusion_matrix, macro_f1_from_confusion  # noqa: E402
from src.models import build_r2plus1d18  # noqa: E402
from src.training import evaluate, train_one_epoch  # noqa: E402


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default="data/UCF-101")
    parser.add_argument("--split-csv", default="splits/split01_seed42.csv")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=positive_int, default=16)
    parser.add_argument("--sampling", choices=["uniform", "contiguous"], default="uniform")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--cpu-threads", type=positive_int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-pretrained", action="store_true", help="offline debugging only")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="evaluate test using the best validation checkpoint")
    parser.add_argument("--out", default="results/metrics/baseline_r2plus1d18.json")
    parser.add_argument("--confusion-out", default="results/metrics/baseline_r2plus1d18_val_confusion.csv")
    parser.add_argument("--predictions-out", default="results/metrics/baseline_r2plus1d18_val_predictions.csv")
    parser.add_argument("--checkpoint-out", default="results/checkpoints/baseline_r2plus1d18_best.pt")
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("num-workers and weight-decay must be nonnegative; lr must be positive")
    return args


def write_json(path, value):
    """Replace metrics atomically so completed epochs survive interruptions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_confusion(path, matrix, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["truth\\pred", *names])
        writer.writerows([name, *map(int, row)] for name, row in zip(names, matrix))


def write_predictions(path, samples, target, pred, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "truth", "prediction", "correct"])
        for (video_path, _), truth, prediction in zip(samples, target.tolist(), pred.tolist()):
            writer.writerow([video_path, names[truth], names[prediction], int(truth == prediction)])


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run outside the agent sandbox with CUDA-enabled PyTorch, or use --device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.cpu_threads)

    rows = read_csv(args.split_csv)
    assert_no_group_overlap(rows)
    required = {"train", "val", "test"} if args.evaluate_test else {"train", "val"}
    check_files_exist([row for row in rows if row["split"] in required], args.video_root)

    train_loader, val_loader, test_loader, names = build_dataloaders(
        video_root=args.video_root, split_csv=args.split_csv, sampling=args.sampling,
        num_frames=args.num_frames, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    model = build_r2plus1d18(len(names), pretrained=not args.no_pretrained,
                            freeze_backbone=not args.unfreeze_backbone).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = Path(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": "r2plus1d_18", "pretrained": not args.no_pretrained,
        "weights": None if args.no_pretrained else "R2Plus1D_18_Weights.KINETICS400_V1",
        "freeze_backbone": not args.unfreeze_backbone, "classes": names,
        "config": vars(args), "device": str(device),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": str(torchvision.__version__),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "split_sha256": hashlib.sha256(Path(args.split_csv).read_bytes()).hexdigest(),
        "split_sizes": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "selection": "highest val_accuracy, then lowest val_loss",
        "checkpoint": str(checkpoint_path), "history": [], "best_val_epoch": None,
        "test": None, "status": "running",
    }
    print(f"device={device}; train={len(train_loader.dataset)}; val={len(val_loader.dataset)}; classes={len(names)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device)
        val_result, pred, target = evaluate(model, val_loader, device)
        matrix = confusion_matrix(target.numpy(), pred.numpy(), len(names))
        row = {
            "epoch": epoch, "train_loss": train_result.loss, "train_accuracy": train_result.accuracy,
            "train_seconds": train_result.seconds, "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy, "val_macro_f1": macro_f1_from_confusion(matrix),
            "val_seconds": val_result.seconds,
        }
        output["history"].append(row)
        best = output["best_val_epoch"]
        if best is None or (row["val_accuracy"], -row["val_loss"]) > (best["val_accuracy"], -best["val_loss"]):
            output["best_val_epoch"] = row
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "classes": names,
                        "config": vars(args), "validation": row, "split_sha256": output["split_sha256"]}, temporary)
            temporary.replace(checkpoint_path)
            write_confusion(args.confusion_out, matrix, names)
            write_predictions(args.predictions_out, val_loader.dataset.samples, target, pred, names)
        write_json(args.out, output)
        print(f"epoch {epoch:02d} | train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.3f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.3f} macro-F1 {row['val_macro_f1']:.3f}", flush=True)

    if args.evaluate_test:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        result, pred, target = evaluate(model, test_loader, device)
        matrix = confusion_matrix(target.numpy(), pred.numpy(), len(names))
        output["test"] = {"loss": result.loss, "accuracy": result.accuracy,
                          "macro_f1": macro_f1_from_confusion(matrix), "seconds": result.seconds,
                          "checkpoint_epoch": checkpoint["epoch"]}
        write_confusion(Path(args.confusion_out).with_name(Path(args.confusion_out).stem + "_test.csv"), matrix, names)
    output["status"] = "completed"
    write_json(args.out, output)
    print(f"Saved metrics: {args.out}; best checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
