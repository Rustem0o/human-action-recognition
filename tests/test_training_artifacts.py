"""Check validation selection, saved weights and the held-out test boundary."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('evaluate_test', [False, True])
def test_best_checkpoint_and_validation_artifacts(tmp_path, monkeypatch, evaluate_test):
    script = load_script('train_baseline')
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    names = ['A', 'B']
    rows = []
    loaders = []
    for split, group in [('train', 'g08'), ('val', 'g09'), ('test', 'g01')]:
        dataset = TensorDataset(torch.randn(2, 3), torch.tensor([0, 1]))
        dataset.samples = [(f'{c}/v_{c}_{group}_c01.avi', i) for i, c in enumerate(names)]
        for path, label in dataset.samples:
            rows.append(dict(path=path, label=names[label], group=group, split=split))
            if split != 'test' or evaluate_test:
                video = tmp_path / path
                video.parent.mkdir(exist_ok=True)
                video.write_bytes(b'fixture; loader is replaced in this test')
        loaders.append(DataLoader(dataset, batch_size=2))
    split_path = tmp_path / 'split.csv'
    with split_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'group', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(script, 'build_dataloaders', lambda **kwargs: (*loaders, names))
    monkeypatch.setattr(script, 'build_r2plus1d18', lambda *args, **kwargs: model)
    snapshots = []
    test_calls = []

    def train(model, loader, optimizer, device):
        with torch.no_grad():
            model.weight.add_(1)
        result = script.train_one_epoch_original(model, loader, optimizer, device)
        snapshots.append(model.weight.detach().clone())
        return result

    script.train_one_epoch_original = script.train_one_epoch
    monkeypatch.setattr(script, 'train_one_epoch', train)

    def evaluate(model, loader, device):
        if loader is loaders[2]:
            test_calls.append(model.weight.detach().clone())
        # First validation epoch wins; second deliberately has worse accuracy.
        predictions = torch.tensor([0, 1]) if len(snapshots) == 1 else torch.tensor([1, 0])
        result = script.evaluate_original(model, loader, device)[0]
        if loader is loaders[1]:
            result.accuracy = 1.0 if len(snapshots) == 1 else 0.0
        return result, predictions, torch.tensor([0, 1])

    script.evaluate_original = script.evaluate
    monkeypatch.setattr(script, 'evaluate', evaluate)
    out = tmp_path / 'metrics.json'
    checkpoint = tmp_path / 'best.pt'
    confusion = tmp_path / 'confusion.csv'
    predictions = tmp_path / 'predictions.csv'
    args = ['--video-root', str(tmp_path), '--split-csv', str(split_path), '--device', 'cpu',
            '--epochs', '2', '--num-workers', '0', '--no-pretrained', '--out', str(out),
            '--checkpoint-out', str(checkpoint), '--confusion-out', str(confusion),
            '--predictions-out', str(predictions)]
    if evaluate_test:
        args.append('--evaluate-test')
    script.main(args)
    metrics = json.loads(out.read_text())
    assert metrics['status'] == 'completed'
    assert metrics['best_val_epoch']['epoch'] == 1
    assert metrics['best_val_epoch']['val_macro_f1'] == 1
    state = torch.load(checkpoint, weights_only=True)
    assert state['epoch'] == 1
    assert torch.equal(state['model_state_dict']['weight'], snapshots[0])
    assert not torch.equal(state['model_state_dict']['weight'], snapshots[1])
    with confusion.open() as f:
        assert list(csv.reader(f)) == [['truth\\pred', 'A', 'B'], ['A', '1', '0'], ['B', '0', '1']]
    with predictions.open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 2
    assert all(record['correct'] == '1' for record in records)
    if evaluate_test:
        assert len(test_calls) == 1
        assert torch.equal(test_calls[0], state['model_state_dict']['weight'])
        assert metrics['test']['checkpoint_epoch'] == 1
    else:
        assert test_calls == []
        assert metrics['test'] is None


def test_zero_epochs_rejected():
    with pytest.raises(SystemExit):
        load_script('train_baseline').parse_args(['--epochs', '0'])
