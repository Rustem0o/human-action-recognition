"""ZIP extraction must follow the existing split and reject incomplete archives."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prepare_ucf101', Path(__file__).parents[1] / 'scripts/prepare_ucf101.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_extracts_only_csv_clips(tmp_path):
    archive = tmp_path / 'videos.zip'
    relative = 'PushUps/v_PushUps_g08_c01.avi'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('UCF-101/' + relative, b'video fixture')
        z.writestr('UCF-101/Other/other.avi', b'ignored')
        z.writestr('../../outside.txt', b'ignored')
    root = tmp_path / 'videos'
    hashes = prepare.extract_subset(archive, root, [{'path': relative}])
    assert (root / relative).read_bytes() == b'video fixture'
    assert set(hashes) == {relative}
    assert not (root / 'Other').exists()
    assert not (tmp_path / 'outside.txt').exists()


def test_missing_clip_fails_before_writing(tmp_path):
    archive = tmp_path / 'videos.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('UCF-101/Other/other.avi', b'ignored')
    with pytest.raises(FileNotFoundError, match='missing'):
        prepare.extract_subset(archive, tmp_path / 'out', [{'path': 'A/a.avi'}])
    assert not (tmp_path / 'out').exists()
