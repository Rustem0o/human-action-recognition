"""Build a group-aware train/val/test split for a 10-class subset of UCF101.

Test  = official testlist01.txt (restricted to our classes).
Val   = ~20% of the TRAIN groups of each class (all clips of those groups).
Train = remaining clips of trainlist01.txt.

A group is identified by (class, gXX): clips of one group come from the same
source video, so a group never appears in more than one split.

Usage:
    python -m src.data.splits --split-dir data/ucfTrainTestlist --out splits/split01_seed42.csv \
        [--video-root data/UCF-101] [--val-fraction 0.2] [--seed 42]
"""

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Folder names exactly as in UCF101 classInd.txt.
# The UCF website calls "Basketball" "Basketball Shooting"; the folder is "Basketball".
CLASSES = [
    "PlayingGuitar",
    "PlayingSitar",
    "FrontCrawl",
    "BreastStroke",
    "Basketball",
    "BasketballDunk",
    "PushUps",
    "BodyWeightSquats",
    "Lunges",
    "JumpingJack",
]

SPLIT_NAMES = ("train", "val", "test")
CSV_FIELDS = ["path", "label", "group", "split"]

# v_<ClassName>_g<XX>_c<YY>.avi
_CLIP_RE = re.compile(r"^v_(?P<cls>.+)_g(?P<group>\d+)_c(?P<clip>\d+)\.avi$", re.IGNORECASE)


def class_names():
    """Label order used everywhere: alphabetical, so label indices are stable."""
    return sorted(CLASSES)


def read_class_index(class_ind_path):
    """Return the class names listed in classInd.txt ("<idx> <ClassName>" per line)."""
    names = []
    with open(class_ind_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                names.append(parts[1])
    return names


def check_classes(class_ind_path, classes=CLASSES):
    available = set(read_class_index(class_ind_path))
    missing = [c for c in classes if c not in available]
    if missing:
        raise ValueError(
            f"Classes not found in {class_ind_path}: {missing}. "
            f"Check the spelling against the folder names in classInd.txt."
        )


def parse_clip(rel_path):
    """'PushUps/v_PushUps_g08_c01.avi' -> (label, group_id) with group_id like 'g08'."""
    rel_path = rel_path.replace("\\", "/")
    folder, _, fname = rel_path.rpartition("/")
    m = _CLIP_RE.match(fname)
    if m is None:
        raise ValueError(f"Unexpected UCF101 file name: {rel_path}")
    return folder, f"g{int(m.group('group')):02d}"


def read_list(list_path, classes):
    """Read trainlist/testlist; keep only clips of `classes`. Returns list of (path, label, group)."""
    keep = set(classes)
    items = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path = line.split()[0].replace("\\", "/")  # trainlist has a trailing label index
            label, group = parse_clip(rel_path)
            if label in keep:
                items.append((rel_path, label, group))
    return items


def build_split(split_dir, val_fraction=0.2, seed=42, classes=CLASSES, split_id=1):
    """Return a list of dict rows with keys path, label, group, split."""
    split_dir = Path(split_dir)
    for name in ("classInd.txt", f"trainlist0{split_id}.txt", f"testlist0{split_id}.txt"):
        if not (split_dir / name).is_file():
            raise FileNotFoundError(f"{name} not found in {split_dir}")
    check_classes(split_dir / "classInd.txt", classes)

    train_items = read_list(split_dir / f"trainlist0{split_id}.txt", classes)
    test_items = read_list(split_dir / f"testlist0{split_id}.txt", classes)

    for c in classes:
        if not any(lbl == c for _, lbl, _ in train_items):
            raise ValueError(f"No training clips found for class {c}")
        if not any(lbl == c for _, lbl, _ in test_items):
            raise ValueError(f"No test clips found for class {c}")

    # Choose validation groups per class (stratified), deterministic for a given seed.
    rng = random.Random(seed)
    groups_per_class = defaultdict(set)
    for _, label, group in train_items:
        groups_per_class[label].add(group)
    val_groups = set()
    for label in sorted(groups_per_class):
        groups = sorted(groups_per_class[label])
        n_val = max(1, round(val_fraction * len(groups)))
        if n_val >= len(groups):
            raise ValueError(f"Class {label} has too few train groups ({len(groups)}) for a val split")
        val_groups.update((label, g) for g in rng.sample(groups, n_val))

    rows = []
    for path, label, group in train_items:
        split = "val" if (label, group) in val_groups else "train"
        rows.append({"path": path, "label": label, "group": group, "split": split})
    for path, label, group in test_items:
        rows.append({"path": path, "label": label, "group": group, "split": "test"})

    assert_no_group_overlap(rows)
    return rows


def assert_no_group_overlap(rows):
    groups = {s: set() for s in SPLIT_NAMES}
    paths = {s: set() for s in SPLIT_NAMES}
    for r in rows:
        groups[r["split"]].add((r["label"], r["group"]))
        paths[r["split"]].add(r["path"])
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            g = groups[a] & groups[b]
            assert not g, f"Group overlap between {a} and {b}: {sorted(g)[:10]}"
            p = paths[a] & paths[b]
            assert not p, f"Clip overlap between {a} and {b}: {sorted(p)[:10]}"


def check_files_exist(rows, video_root):
    video_root = Path(video_root)
    missing = [r["path"] for r in rows if not (video_root / r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clips listed in the split are missing under {video_root}, "
            f"e.g. {missing[:5]}. Did you extract all 10 class folders?"
        )


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_table(rows):
    """Text table: clips per class per split, plus group counts."""
    clips = defaultdict(int)
    groups = defaultdict(set)
    for r in rows:
        clips[(r["label"], r["split"])] += 1
        groups[(r["label"], r["split"])].add(r["group"])

    labels = sorted({r["label"] for r in rows})
    w = max(len("TOTAL"), *(len(l) for l in labels))
    header = f"{'class':<{w}} | " + " | ".join(f"{s + ' clips':>11} {s + ' groups':>12}" for s in SPLIT_NAMES)
    lines = [header, "-" * len(header)]
    totals = defaultdict(int)
    for label in labels:
        cells = []
        for s in SPLIT_NAMES:
            n_c, n_g = clips[(label, s)], len(groups[(label, s)])
            totals[(s, "c")] += n_c
            totals[(s, "g")] += n_g
            cells.append(f"{n_c:>11} {n_g:>12}")
        lines.append(f"{label:<{w}} | " + " | ".join(cells))
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<{w}} | " + " | ".join(f"{totals[(s, 'c')]:>11} {totals[(s, 'g')]:>12}" for s in SPLIT_NAMES)
    )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", required=True, help="folder with classInd.txt, trainlist01.txt, testlist01.txt")
    p.add_argument("--out", default="splits/split01_seed42.csv", help="output CSV path")
    p.add_argument("--video-root", default=None, help="optional: UCF-101 folder; checks that every clip exists")
    p.add_argument("--val-fraction", type=float, default=0.2, help="fraction of train groups per class for val")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rows = build_split(args.split_dir, val_fraction=args.val_fraction, seed=args.seed)
    if args.video_root:
        check_files_exist(rows, args.video_root)
        print(f"All {len(rows)} clips found under {args.video_root}")
    write_csv(rows, args.out)
    print(f"Saved {len(rows)} rows to {args.out}  (seed={args.seed}, val_fraction={args.val_fraction})")
    print("No group overlap between train / val / test: OK")
    print(summary_table(rows))


if __name__ == "__main__":
    sys.exit(main())
