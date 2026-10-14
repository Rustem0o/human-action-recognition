"""Download a pinned UCF101 ZIP mirror and extract only the committed subset.

The committed CSV defines train/validation/test; the mirror's splits are never used.
Only matching class/filename paths are written, without ZipFile.extractall().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import assert_no_group_overlap, check_files_exist, read_csv  # noqa: E402

DATASET_REVISION = "04d4e5ca1dc93606cb58752b0c08331e598743a4"
DEFAULT_URL = f"https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/{DATASET_REVISION}/UCF-101.zip"
ARCHIVE_SHA256 = "eb77e54dfafd9c77e7b086f5e5ef7738cabd16a701b3f5c301a1f6c81fc4756e"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    """Resume partial transfers; restart safely when the server ignores Range."""
    destination = Path(destination)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "human-action-recognition/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Download returned an unexpected Content-Range")
        else:
            offset = 0
        remaining = int(response.headers.get("Content-Length", 0))
        received = 0
        last_print = time.monotonic()
        with partial.open("ab" if offset else "wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block)
                received += len(block)
                if time.monotonic() - last_print >= 10:
                    total = f" / {(offset + remaining) / 2**30:.2f} GiB" if remaining else ""
                    print(f"Downloaded {(offset + received) / 2**30:.2f} GiB{total}", flush=True)
                    last_print = time.monotonic()
        if remaining and received != remaining:
            raise RuntimeError("Download was interrupted; run the command again to resume")
    partial.replace(destination)


def extract_subset(archive_path, video_root, rows):
    """Write only regular ZIP members corresponding to exact CSV paths."""
    wanted = {row["path"] for row in rows}
    for name in wanted:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
            raise ValueError(f"Invalid subset path: {name}")
    root = Path(video_root)
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for item in archive.infolist():
            parts = PurePosixPath(item.filename).parts
            if item.is_dir() or len(parts) < 2:
                continue
            relative = "/".join(parts[-2:])
            if relative in wanted:
                if relative in members:
                    raise ValueError(f"Duplicate clip in ZIP: {relative}")
                members[relative] = item
        missing = wanted - members.keys()
        if missing:
            raise FileNotFoundError(f"Archive is missing {len(missing)} subset clips: {sorted(missing)[:5]}")
        for index, (relative, member) in enumerate(sorted(members.items()), 1):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Output path escapes video root: {relative}")
            temporary = target.with_suffix(".avi.part")
            with archive.open(member) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            temporary.replace(target)
            hashes[relative] = sha256(target)
            if index % 100 == 0:
                print(f"Extracted {index}/{len(members)} clips", flush=True)
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", default="splits/split01_seed42.csv")
    parser.add_argument("--video-root", default="data/UCF-101")
    parser.add_argument("--archive", default="data/UCF-101.zip")
    args = parser.parse_args(argv)
    rows = read_csv(args.split_csv)
    assert_no_group_overlap(rows)
    manifest_path = Path(args.video_root).parent / "ucf101_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {row["path"] for row in rows}
        if (manifest.get("split_sha256") == sha256(args.split_csv)
                and expected == set(manifest.get("video_sha256", {}))
                and all((Path(args.video_root) / p).is_file()
                        and sha256(Path(args.video_root) / p) == manifest["video_sha256"][p] for p in expected)):
            print(f"All {len(rows)} clips verified against the local manifest; no download needed.", flush=True)
            return
    print(f"Dataset source: {DEFAULT_URL}", flush=True)
    download(DEFAULT_URL, args.archive)
    archive_digest = sha256(args.archive)
    if archive_digest != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA256 mismatch: {archive_digest}. Expected {ARCHIVE_SHA256}")
    hashes = extract_subset(args.archive, args.video_root, rows)
    check_files_exist(rows, args.video_root)
    manifest = {"source": DEFAULT_URL, "revision": DATASET_REVISION,
                "archive_sha256": archive_digest, "split_sha256": sha256(args.split_csv),
                "video_sha256": hashes, "clips": len(rows)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {len(rows)} real UCF101 clips in {args.video_root}", flush=True)


if __name__ == "__main__":
    main()
