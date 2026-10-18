# human-action-recognition
Recognizing human actions in short video clips with pretrained models.

We compare two Kinetics-400 pretrained video models from torchvision, R(2+1)D-18 (baseline) and
Video Swin-T, on a 10-class subset of UCF101 (official split 1).

This README covers the **data pipeline** (split, frame sampling, transforms, DataLoaders)
and the first working **R(2+1)D-18 baseline** for the 13 Oct checkpoint.

## Classes

Folder names exactly as in UCF101 `classInd.txt`:

| | | | | |
|---|---|---|---|---|
| PlayingGuitar | PlayingSitar | FrontCrawl | BreastStroke | Basketball |
| BasketballDunk | PushUps | BodyWeightSquats | Lunges | JumpingJack |

The UCF website calls `Basketball` "Basketball Shooting"; in the archive and in `classInd.txt` the
folder is `Basketball`. Label indices are the **alphabetical** order of these names
(`class_names` returned by `build_dataloaders`).

## Repository structure

```
src/
  data/
    splits.py          # builds the group-aware train/val/test split CSV
    dataset.py         # video decoding, frame sampling, transforms, build_dataloaders()
  models/ training/ evaluation/ utils/   # next stages
configs/ notebooks/ reports/
scripts/
  visualize_batch.py   # saves a figure of one batch + prints average loading time per batch
  train_baseline.py    # trains/evaluates the first R(2+1)D-18 baseline
tests/
  test_splits.py       # split logic on fake split files
  test_data_smoke.py   # loader shapes on tiny synthetic .avi files (tests only)
splits/                # generated split CSVs (small, committed)
data/                  # UCF101 videos and split files (NOT committed, see .gitignore)
results/               # figures/, metrics/; results/checkpoints/ is ignored
requirements.txt
```

## Split

- **Test** = official `testlist01.txt` restricted to the 10 classes (groups g01-g07).
- **Validation** = for each class, `round(0.2 * 18) = 4` of its 18 train groups (g08-g25), picked
  with `random.Random(seed)` (default seed 42); **all** clips of those groups go to val.
- **Train** = the remaining clips of `trainlist01.txt`.
- A group is identified by (class, gXX). Clips of one group come from the same source video, so the
  script asserts that no group (and no clip) appears in two splits.

CSV columns: `path` (relative to the `UCF-101` folder), `label` (class name), `group` (e.g. `g08`),
`split` (`train` / `val` / `test`).

## Frame sampling and transforms

- Decoding: OpenCV (`opencv-python-headless`).
- `sampling="uniform"`: `num_frames` evenly spaced frames over the whole clip (default 16).
- `sampling="contiguous"`: one window of `num_frames` consecutive frames; random start for train,
  centered for val/test.
- Clips shorter than `num_frames`: all frames, then the last frame is repeated.
- Unreadable videos raise `RuntimeError` naming the file.
- Resize size, crop size, mean and std come from
  `torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1.transforms()` (resize to 128x171,
  crop 112x112). Train: random crop + horizontal flip (same for all frames of a clip).
  Val/test: center crop only.
- Output: `videos` float32 `(B, C, T, H, W)`, normalized; `labels` int64 `(B,)`.

## Run on Google Colab / Kaggle

Use a GPU runtime. On Kaggle, turn on **Internet** in the notebook settings first.
In a Colab cell (on Kaggle replace `/content` with `/kaggle/working`):

```bash
%cd /content
!git clone https://github.com/Rustem0o/human-action-recognition.git
%cd human-action-recognition
!pip install -q -r requirements.txt
```

Download UCF101 videos (~6.5 GB .rar) and the official split files. The UCF server sometimes has an
invalid SSL certificate, hence `--no-check-certificate`:

```bash
!mkdir -p data
!wget -q --show-progress --no-check-certificate -P data https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
!wget -q --show-progress --no-check-certificate -P data https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
!unzip -q -o data/UCF101TrainTestSplits-RecognitionTask.zip -d data   # -> data/ucfTrainTestlist/
```

Extract only the 10 class folders (the archive is .rar, so `unrar` is needed; if it cannot be
installed, `unar` works too):

```bash
%%bash
which unrar || apt-get -qq install -y unrar || apt-get -qq install -y unar
cd data
for c in PlayingGuitar PlayingSitar FrontCrawl BreastStroke Basketball BasketballDunk PushUps BodyWeightSquats Lunges JumpingJack; do
  if which unrar > /dev/null; then unrar x -o+ -idq UCF101.rar "UCF-101/$c/*" ./
  else unar -q -f -D UCF101.rar "UCF-101/$c/*"; fi
done
ls UCF-101            # must list exactly the 10 folders
rm UCF101.rar         # optional, frees ~6.5 GB
```

Result: `data/UCF-101/<ClassName>/v_<ClassName>_gXX_cYY.avi` and `data/ucfTrainTestlist/`.

### 1. Build the split

```bash
!python -m src.data.splits --split-dir data/ucfTrainTestlist --video-root data/UCF-101 --out splits/split01_seed42.csv --seed 42
```

Prints a table of clips and groups per class per split and fails if a class is missing from
`classInd.txt`, if any listed clip is missing on disk, or if groups overlap between splits.

### 2. Run the tests

```bash
!python -m pytest -q
```

The smoke test writes a few tiny synthetic .avi files to a temporary folder; they are only used to
check tensor shapes and are never used for reported results.

### 3. Visualize a batch and measure loading time

```bash
!python scripts/visualize_batch.py --video-root data/UCF-101 --split-csv splits/split01_seed42.csv --split train --sampling uniform --out results/figures/batch_train_uniform.png
!python scripts/visualize_batch.py --video-root data/UCF-101 --split-csv splits/split01_seed42.csv --split val --sampling contiguous --out results/figures/batch_val_contiguous.png
```

### 4. Train the first baseline

The first checkpoint baseline uses torchvision's R(2+1)D-18 video classifier.
By default, it loads Kinetics-400 pretrained weights, replaces the classifier
with a 10-class head, freezes the backbone, and trains only the final layer.

```bash
!python scripts/train_baseline.py \
  --video-root data/UCF-101 \
  --split-csv splits/split01_seed42.csv \
  --epochs 3 \
  --batch-size 8 \
  --num-workers 2 \
  --out results/metrics/baseline_r2plus1d18.json
```

The script saves JSON metrics and a confusion matrix CSV in `results/metrics/`.
Use `--no-pretrained` only for offline debugging when pretrained weights cannot
be downloaded.

## 13 Oct checkpoint

See `reports/progress_2026-10-13.md` for the first checkpoint summary: data
pipeline, split, baseline configuration, initial checks, and current blocker.

## Using the DataLoaders

```python
from src.data import build_dataloaders

train_loader, val_loader, test_loader, class_names = build_dataloaders(
    video_root="data/UCF-101",
    split_csv="splits/split01_seed42.csv",
    sampling="uniform",      # or "contiguous"
    num_frames=16,
    batch_size=8,
    num_workers=2,
    seed=42,
)
for videos, labels in train_loader:   # videos: (B, 3, 16, 112, 112) float32, labels: (B,) int64
    ...
```

- Only `train_loader` is shuffled. Val/test keep CSV order, so `test_loader.dataset.samples[i]`
  is the `(path, label_idx)` of the i-th test prediction (useful for failure analysis).
- `build_dataloaders` seeds `random`, `numpy` and `torch`, sets cuDNN to deterministic mode, and
  seeds the DataLoader generator and workers. Random crops/flips/windows are reproducible for the
  same `seed` **and** `num_workers`.
- For Video Swin-T, pass its preset so preprocessing matches that model:
  `build_dataloaders(..., weights=torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1)`.
  The default (`weights=None`) uses the R(2+1)D-18 preset.


## Baseline results

Validation on 214 videos from the 10-class UCF101 subset, with a frozen
Kinetics-400 pretrained R(2+1)D-18 backbone and a trained classifier head.

| Training epochs | Selected epoch | Validation accuracy | Macro-F1 | Validation loss |
|---:|---:|---:|---:|---:|
| [3](results/metrics/baseline_r2plus1d18.json) | 3 | 95.79% (205/214) | 0.9531 | 0.2723 |
| [10](results/metrics/baseline_r2plus1d18_10epochs.json) | 9 | 95.79% (205/214) | 0.9529 | 0.1769 |

The test set was not evaluated.

[Learning curves](results/figures/baseline_r2plus1d18_10epochs_learning_curves.png)
· [Validation confusion matrix](results/figures/baseline_r2plus1d18_10epochs_val_confusion.png)


## Frame sampling comparison

Frozen Kinetics-400 pretrained R(2+1)D-18 backbone, 10 training epochs, batch size 4,
learning rate 0.001, weight decay 0.0001, seed 42; the same 712 train / 214 validation clips.
Uniform sampling spans the full video. Contiguous sampling uses a random consecutive
window for training and a centered window for validation.

| Frames | Sampling | Selected epoch | Validation accuracy | Macro-F1 | Validation loss |
|---:|---|---:|---:|---:|---:|
| 8 | uniform | 5 | 94.39% | 0.9391 | 0.2390 |
| 16 | uniform | 9 | 95.79% | 0.9529 | 0.1769 |
| 32 | uniform | 7 | 97.20% | 0.9681 | 0.1542 |
| 8 | contiguous | 9 | 92.52% | 0.9198 | 0.2183 |
| 16 | contiguous | 9 | 93.93% | 0.9305 | 0.1943 |
| 32 | contiguous | 3 | 95.33% | 0.9456 | 0.2642 |

Uniform sampling improved accuracy by 1.87 percentage points over contiguous sampling
at each frame count. Increasing uniform sampling from 16 to 32 frames added 1.40
percentage points, with 2.16 times the measured training time.

Selected: **32 frames, uniform** — **97.20% (208/214)** accuracy,
macro-F1 **0.9681**, validation loss **0.1542**.
Selection uses highest validation accuracy, then lowest validation loss. The test set was not evaluated.

[Per-run configurations](configs/baseline_frames/) · [Logs, metrics and predictions](results/experiments/baseline_frames/) ·
[Summary](results/experiments/baseline_frames/summary.csv) · [Comparison plot](results/figures/baseline_frames/comparison.png) ·
[Selected validation confusion matrix](results/figures/baseline_frames/best_val_confusion.png)
