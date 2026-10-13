# video-action-recognition
Recognizing human actions in short video clips with pretrained models.

We compare two Kinetics-400 pretrained video models from torchvision, R(2+1)D-18 (baseline) and
Video Swin-T, on a 10-class subset of UCF101 (official split 1).

This README currently covers the **data pipeline** (split, frame sampling, transforms, DataLoaders).

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
!git clone https://github.com/ketrinekrutova/video-action-recognition.git
%cd video-action-recognition
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
