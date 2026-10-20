#!/usr/bin/env python3
"""
Interactive comparison tool for experiment result JSON files.

Usage:
    python scripts/compare_results.py
"""

import csv
import json
from pathlib import Path


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_validation(result):
    best = result.get("best_val_epoch")
    if best:
        return best

    history = result.get("history", [])
    if not history:
        return {}

    return max(
        history,
        key=lambda x: (x.get("val_accuracy", float("-inf")),
                       -x.get("val_loss", float("inf")))
    )


def main():
    print("\n=== Experiment Results Comparison ===\n")

    while True:
        raw = input("How many result files do you want to compare? ").strip()
        try:
            count = int(raw)
            if count >= 2:
                break
        except ValueError:
            pass
        print("Please enter an integer >= 2.")

    results = []

    for i in range(1, count + 1):
        print(f"\n--- File {i} ---")

        while True:
            path_text = input("Path to JSON result file: ").strip()
            path = Path(path_text).expanduser()

            if path.is_file():
                break

            print(f"File not found: {path}")

        default_name = path.stem
        name = ask("Name for this experiment", default_name)

        result = load_result(path)
        results.append((name, path, result))

    default_output = "results/metrics/model_comparison.csv"
    output_text = ask("\nOutput CSV path", default_output)
    output = Path(output_text).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, path, result in results:
        best = best_validation(result)
        config = result.get("config", {})
        test = result.get("test") or {}

        frames = config.get("num_frames", "")
        sampling = config.get("sampling", "")

        # Some result files may store the model name under "model".
        model = result.get("model", "")

        rows.append({
            "name": name,
            "model": model,
            "frames": frames,
            "sampling": sampling,
            "epochs": config.get("epochs", ""),
            "best_epoch": best.get("epoch", ""),
            "val_accuracy": best.get("val_accuracy", ""),
            "val_macro_f1": best.get("val_macro_f1", ""),
            "val_loss": best.get("val_loss", ""),
            "train_seconds": sum(
                x.get("train_seconds", 0)
                for x in result.get("history", [])
            ),
            "test_accuracy": test.get("accuracy", ""),
            "test_macro_f1": test.get("macro_f1", ""),
            "test_loss": test.get("loss", ""),
            "test_seconds": test.get("seconds", ""),
            "source_file": str(path),
        })

    fieldnames = list(rows[0].keys())

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Comparison ===\n")

    display_columns = [
        ("name", "Experiment"),
        ("model", "Model"),
        ("frames", "Frames"),
        ("sampling", "Sampling"),
        ("best_epoch", "Best epoch"),
        ("val_accuracy", "Val accuracy"),
        ("val_macro_f1", "Val macro-F1"),
        ("val_loss", "Val loss"),
        ("test_accuracy", "Test accuracy"),
        ("test_macro_f1", "Test macro-F1"),
    ]

    for _, title in display_columns:
        print(f"{title:18}", end="")
    print()

    print("-" * (18 * len(display_columns)))

    for row in rows:
        for key, _ in display_columns:
            value = row[key]

            if key in {"val_accuracy", "val_macro_f1",
                       "test_accuracy", "test_macro_f1"}:
                if value != "":
                    value = f"{float(value) * 100:.2f}%"

            elif key == "val_loss" and value != "":
                value = f"{float(value):.4f}"

            print(f"{str(value):18}", end="")
        print()

    print(f"\nSaved comparison: {output}")


if __name__ == "__main__":
    main()
