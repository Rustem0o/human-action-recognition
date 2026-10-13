"""Tests for src/data/splits.py on small fake split files (no real data needed)."""

import pytest

from src.data.splits import CLASSES, build_split, parse_clip, read_csv, summary_table, write_csv


def make_split_dir(tmp_path, classes=CLASSES, extra_classes=("ApplyEyeMakeup",), clips_per_group=4):
    """Fake ucfTrainTestlist: groups 1-7 test, 8-25 train (as in official split 1), CRLF endings."""
    all_classes = list(extra_classes) + list(classes)
    (tmp_path / "classInd.txt").write_text(
        "".join(f"{i + 1} {c}\r\n" for i, c in enumerate(all_classes)), encoding="utf-8")
    train, test = [], []
    for i, c in enumerate(all_classes):
        for g in range(1, 26):
            for k in range(1, clips_per_group + 1):
                path = f"{c}/v_{c}_g{g:02d}_c{k:02d}.avi"
                if g <= 7:
                    test.append(f"{path}\r\n")
                else:
                    train.append(f"{path} {i + 1}\r\n")
    (tmp_path / "trainlist01.txt").write_text("".join(train), encoding="utf-8")
    (tmp_path / "testlist01.txt").write_text("".join(test), encoding="utf-8")
    return tmp_path


def test_parse_clip():
    assert parse_clip("PushUps/v_PushUps_g08_c01.avi") == ("PushUps", "g08")


def test_split_properties(tmp_path):
    rows = build_split(make_split_dir(tmp_path), val_fraction=0.2, seed=42)
    assert {r["label"] for r in rows} == set(CLASSES)  # extra class filtered out
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    assert all(by_split.values())

    # test = official test list (groups 1-7) only
    assert {int(r["group"][1:]) for r in by_split["test"]} == set(range(1, 8))

    # val = round(0.2 * 18) = 4 groups per class, with all of their clips
    for c in CLASSES:
        val_groups = {r["group"] for r in by_split["val"] if r["label"] == c}
        train_groups = {r["group"] for r in by_split["train"] if r["label"] == c}
        assert len(val_groups) == 4
        assert not val_groups & train_groups
        assert len(val_groups | train_groups) == 18

    groups = {s: {(r["label"], r["group"]) for r in v} for s, v in by_split.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_split_is_deterministic(tmp_path):
    d = make_split_dir(tmp_path)
    assert build_split(d, seed=42) == build_split(d, seed=42)
    val42 = {(r["label"], r["group"]) for r in build_split(d, seed=42) if r["split"] == "val"}
    val7 = {(r["label"], r["group"]) for r in build_split(d, seed=7) if r["split"] == "val"}
    assert val42 != val7


def test_missing_class_raises(tmp_path):
    d = make_split_dir(tmp_path, classes=[c for c in CLASSES if c != "Lunges"])
    with pytest.raises(ValueError, match="Lunges"):
        build_split(d)


def test_csv_roundtrip_and_table(tmp_path):
    rows = build_split(make_split_dir(tmp_path))
    out = tmp_path / "out" / "split.csv"
    write_csv(rows, out)
    assert read_csv(out) == rows
    table = summary_table(rows)
    assert "TOTAL" in table and "PushUps" in table
