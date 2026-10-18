"""Run the six frame-count/sampling configurations and rank validation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_baseline import parse_args as training_args, write_json  # noqa: E402
from src.data import get_preset  # noqa: E402
from src.data.splits import assert_no_group_overlap, read_csv  # noqa: E402

CONFIG_ROOT = Path("configs/baseline_frames")
RESULT_ROOT = Path("results/experiments/baseline_frames")
SELECTION = "highest validation accuracy, then lowest validation loss"
SOURCE_FILES = [
    "scripts/run_frame_ablation.py", "scripts/train_baseline.py",
    "src/data/dataset.py", "src/data/splits.py", "src/models/video_classifiers.py",
    "src/training/engine.py", "src/evaluation/metrics.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def arguments(config):
    """Resolve the same CLI arguments used by training, with no test evaluation."""
    values = config["args"]
    if values.get("evaluate_test", False):
        raise ValueError("Test evaluation is forbidden in the frame ablation")
    if values.get("no_pretrained", False) or values.get("unfreeze_backbone", False):
        raise ValueError("The frame ablation requires the frozen pretrained backbone")
    argv = []
    for name, value in values.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(option)
        elif value is not None:
            argv.extend([option, str(value)])
    resolved = vars(training_args(argv))
    if resolved != values:
        raise ValueError("Config must explicitly record all resolved training arguments")
    return argv


def make_config(frames, sampling):
    run_id = f"{sampling}_f{frames:02d}_seed42"
    out = RESULT_ROOT / run_id
    args = vars(training_args([
        "--num-frames", str(frames), "--sampling", sampling,
        "--epochs", "10", "--batch-size", "4", "--lr", "0.001",
        "--weight-decay", "0.0001", "--seed", "42", "--device", "cuda",
        "--num-workers", "2", "--cpu-threads", "8",
        "--out", str(out / "metrics.json"),
        "--confusion-out", str(out / "val_confusion.csv"),
        "--predictions-out", str(out / "val_predictions.csv"),
        "--checkpoint-out", f"results/checkpoints/baseline_frames/{run_id}_best.pt",
    ]))
    rows = read_csv(ROOT / args["split_csv"])
    assert_no_group_overlap(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    return {
        "run_id": run_id,
        "args": args,
        "model": "r2plus1d_18",
        "weights": "R2Plus1D_18_Weights.KINETICS400_V1",
        "trainable": "fc only; backbone parameters and BatchNorm statistics frozen",
        "loss": "CrossEntropyLoss",
        "optimizer": {"name": "AdamW", "lr": args["lr"], "weight_decay": args["weight_decay"],
                      "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
        "preprocessing": {
            **get_preset(),
            "train_spatial": "random crop and horizontal flip, shared across frames",
            "validation_spatial": "center crop",
            "train_temporal": "random consecutive window" if sampling == "contiguous" else "evenly spaced across full clip",
            "validation_temporal": "centered consecutive window" if sampling == "contiguous" else "evenly spaced across full clip",
            "short_clip_padding": "repeat last frame",
            "resize_interpolation": "bilinear, antialias=True, float input in [0,1]",
        },
        "data": {
            "split_sha256": digest(ROOT / args["split_csv"]),
            "train_clips": sum(r["split"] == "train" for r in rows),
            "validation_clips": sum(r["split"] == "val" for r in rows),
            "test_evaluated": False,
        },
        "selection": SELECTION,
        "source_base_commit": commit,
        "source_sha256": source_fingerprint(),
        "log": str(out / "training.log"),
    }


def prepare_configs():
    paths = []
    for sampling in ("uniform", "contiguous"):
        for frames in (8, 16, 32):
            config = make_config(frames, sampling)
            path = CONFIG_ROOT / (config["run_id"] + ".json")
            if (ROOT / path).exists():
                existing = json.loads((ROOT / path).read_text())
                # Preserve the actual pre-run source provenance across later commits.
                for field in ("source_base_commit", "source_sha256"):
                    config[field] = existing[field]
                if existing != config:
                    raise ValueError(f"Existing config differs: {path}")
            else:
                write_json(ROOT / path, config)
            paths.append(path)
    return paths


def validate_result(config, metrics):
    if metrics["status"] != "completed":
        raise ValueError(f"Run is incomplete: {config['run_id']}")
    if metrics["config"] != config["args"]:
        raise ValueError(f"Metrics/config mismatch: {config['run_id']}")
    if metrics["test"] is not None or config["args"]["evaluate_test"]:
        raise ValueError("Test results must not be used")
    if metrics["split_sha256"] != config["data"]["split_sha256"]:
        raise ValueError("Split fingerprint mismatch")
    history = metrics["history"]
    if [row["epoch"] for row in history] != list(range(1, config["args"]["epochs"] + 1)):
        raise ValueError("Incomplete epoch history")
    best = max(history, key=lambda r: (r["val_accuracy"], -r["val_loss"]))
    if best != metrics["best_val_epoch"]:
        raise ValueError("Incorrect validation checkpoint selection")


def run_config(path, python):
    config = json.loads((ROOT / path).read_text())
    argv = arguments(config)
    if config["source_sha256"] != source_fingerprint():
        raise ValueError(f"Training source changed after config creation: {path}")
    if digest(ROOT / config["args"]["split_csv"]) != config["data"]["split_sha256"]:
        raise ValueError("Dataset split changed")
    out = ROOT / config["args"]["out"]
    if out.exists():
        metrics = json.loads(out.read_text())
        if metrics.get("status") == "completed":
            validate_result(config, metrics)
            if not (ROOT / config["log"]).is_file():
                raise ValueError("Completed run has no saved log")
            print(f"Already complete: {config['run_id']}", flush=True)
            return
        raise ValueError(f"Incomplete run artifacts already exist: {out}. Preserve or move them before retrying.")
    env = {**os.environ, "PYTHONHASHSEED": str(config["args"]["seed"])}
    command = [str(python), "-u", "scripts/train_baseline.py", *argv]
    log_path = ROOT / config["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {config['run_id']}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + shlex.join(command) + "\nCONFIG:\n")
        log.write(json.dumps(config, indent=2) + "\nOUTPUT:\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(f"[{config['run_id']}] {line}", end="", flush=True)
            code = proc.wait()
            log.write(f"EXIT_CODE: {code}\n")
    if code:
        raise RuntimeError(f"Training failed ({code}); see {config['log']}")
    validate_result(config, json.loads(out.read_text()))


def summarize(paths):
    configs = [json.loads((ROOT / p).read_text()) for p in paths]
    # Ensure only frame count / temporal sampling and artifact destinations vary.
    ignored = {"num_frames", "sampling", "out", "confusion_out", "predictions_out", "checkpoint_out"}
    controls = [{k: v for k, v in c["args"].items() if k not in ignored} for c in configs]
    if not all(c == controls[0] for c in controls):
        raise ValueError("The experiment contains confounded training parameters")
    rows = []
    for path, config in zip(paths, configs):
        metrics = json.loads((ROOT / config["args"]["out"]).read_text())
        validate_result(config, metrics)
        best = metrics["best_val_epoch"]
        rows.append({
            "run_id": config["run_id"], "num_frames": config["args"]["num_frames"],
            "sampling": config["args"]["sampling"], "epochs": config["args"]["epochs"],
            "batch_size": config["args"]["batch_size"], "lr": config["args"]["lr"],
            "weight_decay": config["args"]["weight_decay"], "seed": config["args"]["seed"],
            "best_epoch": best["epoch"], "val_accuracy": best["val_accuracy"],
            "val_macro_f1": best["val_macro_f1"], "val_loss": best["val_loss"],
            "train_seconds": sum(r["train_seconds"] for r in metrics["history"]),
            "val_seconds": sum(r["val_seconds"] for r in metrics["history"]),
            "config": str(path), "metrics": config["args"]["out"], "log": config["log"],
            "checkpoint": config["args"]["checkpoint_out"],
        })
    rows.sort(key=lambda r: (-r["val_accuracy"], r["val_loss"], r["run_id"]))
    summary = {"selection": SELECTION, "test_evaluated": False,
               "best_run": rows[0], "runs": rows}
    write_json(ROOT / RESULT_ROOT / "summary.json", summary)
    with (ROOT / RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    paths = prepare_configs()
    if args.prepare_only:
        print("\n".join(map(str, paths)))
        return
    if not args.summarize_only:
        for path in paths:
            run_config(path, args.python)
    summarize(paths)


if __name__ == "__main__":
    main()
