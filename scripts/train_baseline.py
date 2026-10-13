"""Train and evaluate the first working baseline.

Example:
    python scripts/train_baseline.py \
        --video-root data/UCF-101 \
        --split-csv splits/split01_seed42.csv \
        --epochs 3 \
        --batch-size 8 \
        --num-workers 2 \
        --out results/metrics/baseline_r2plus1d18.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dataloaders  # noqa: E402
from src.evaluation import accuracy, confusion_matrix, macro_f1_from_confusion  # noqa: E402
from src.models import build_r2plus1d18  # noqa: E402
from src.training import evaluate, train_one_epoch  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-root", default="data/UCF-101")
    parser.add_argument("--split-csv", default="splits/split01_seed42.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sampling", choices=["uniform", "contiguous"], default="uniform")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="avoid downloading Kinetics weights")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="train all model parameters")
    parser.add_argument("--out", default="results/metrics/baseline_r2plus1d18.json")
    parser.add_argument("--confusion-out", default="results/metrics/baseline_r2plus1d18_confusion.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        video_root=args.video_root,
        split_csv=args.split_csv,
        sampling=args.sampling,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_r2plus1d18(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val = None
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device)
        val_result, _, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_seconds": train_result.seconds,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_seconds": val_result.seconds,
        }
        history.append(row)
        best_val = row if best_val is None or row["val_accuracy"] > best_val["val_accuracy"] else best_val
        print(
            f"epoch {epoch:02d} | train loss {train_result.loss:.4f} acc {train_result.accuracy:.3f} | "
            f"val loss {val_result.loss:.4f} acc {val_result.accuracy:.3f}"
        )

    test_result, test_pred, test_true = evaluate(model, test_loader, device)
    matrix = confusion_matrix(test_true.numpy(), test_pred.numpy(), len(class_names))
    macro_f1 = macro_f1_from_confusion(matrix)

    output = {
        "model": "r2plus1d_18",
        "pretrained": not args.no_pretrained,
        "freeze_backbone": not args.unfreeze_backbone,
        "classes": class_names,
        "config": vars(args),
        "device": str(device),
        "history": history,
        "best_val_epoch": best_val,
        "test": {
            "loss": test_result.loss,
            "accuracy": accuracy(test_true.numpy(), test_pred.numpy()),
            "macro_f1": macro_f1,
            "seconds": test_result.seconds,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    confusion_path = Path(args.confusion_out)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(["truth\\pred"] + class_names)
    rows = [header]
    for name, row in zip(class_names, matrix):
        rows.append(",".join([name] + [str(int(v)) for v in row]))
    confusion_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"saved metrics: {out_path}")
    print(f"saved confusion matrix: {confusion_path}")


if __name__ == "__main__":
    main()
