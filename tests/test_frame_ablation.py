"""Guard comparable frame experiments, validation-only selection and saved logs."""

import json

import pytest

from scripts import run_frame_ablation as ablation


def test_six_configs_change_only_frames_sampling_and_artifact_paths():
    configs = [ablation.make_config(f, s) for s in ('uniform', 'contiguous') for f in (8, 16, 32)]
    assert {(c['args']['num_frames'], c['args']['sampling']) for c in configs} == {
        (f, s) for s in ('uniform', 'contiguous') for f in (8, 16, 32)
    }
    exclude = {'num_frames', 'sampling', 'out', 'confusion_out', 'predictions_out', 'checkpoint_out'}
    controls = [{k: v for k, v in c['args'].items() if k not in exclude} for c in configs]
    assert all(c == controls[0] for c in controls)
    for config in configs:
        assert config['args']['epochs'] == 10
        assert config['args']['batch_size'] == 4
        assert config['args']['lr'] == 0.001
        assert config['args']['seed'] == 42
        assert config['args']['evaluate_test'] is False
        assert config['data']['train_clips'] == 712
        assert config['data']['validation_clips'] == 214
        ablation.arguments(config)


def test_test_evaluation_is_rejected():
    config = ablation.make_config(8, 'uniform')
    config['args']['evaluate_test'] = True
    with pytest.raises(ValueError, match='Test evaluation is forbidden'):
        ablation.arguments(config)


def completed_metrics(config, accuracy=0.8, loss=0.4):
    history = [dict(epoch=i, val_accuracy=accuracy, val_loss=loss + (10-i)*0.01,
                    val_macro_f1=accuracy-0.01, train_seconds=1.0, val_seconds=0.2)
               for i in range(1, 11)]
    return dict(status='completed', config=config['args'], test=None,
                split_sha256=config['data']['split_sha256'], history=history,
                best_val_epoch=history[-1])


def test_wrong_best_epoch_or_incomplete_history_is_rejected():
    config = ablation.make_config(8, 'uniform')
    metrics = completed_metrics(config)
    ablation.validate_result(config, metrics)
    metrics['best_val_epoch'] = metrics['history'][0]
    with pytest.raises(ValueError, match='Incorrect validation checkpoint'):
        ablation.validate_result(config, metrics)
    metrics = completed_metrics(config)
    metrics['history'].pop()
    with pytest.raises(ValueError, match='Incomplete epoch history'):
        ablation.validate_result(config, metrics)


def test_summary_ranks_validation_accuracy_then_loss(tmp_path, monkeypatch):
    configs = [ablation.make_config(f, 'uniform') for f in (8, 16, 32)]
    paths = []
    for index, config in enumerate(configs):
        path = tmp_path / f'config_{index}.json'
        path.write_text(json.dumps(config))
        paths.append(path)
        accuracy, loss = [(0.8, 0.3), (0.9, 0.4), (0.9, 0.2)][index]
        out = tmp_path / config['args']['out']
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(completed_metrics(config, accuracy, loss)))
    monkeypatch.setattr(ablation, 'ROOT', tmp_path)
    result = ablation.summarize(paths)
    assert result['best_run']['num_frames'] == 32
    assert result['test_evaluated'] is False
    assert (tmp_path / ablation.RESULT_ROOT / 'summary.csv').is_file()


def test_test_metrics_cannot_enter_summary():
    config = ablation.make_config(8, 'uniform')
    metrics = completed_metrics(config)
    metrics['test'] = {'accuracy': 1.0}
    with pytest.raises(ValueError, match='Test results must not be used'):
        ablation.validate_result(config, metrics)


def test_training_failure_keeps_config_output_and_exit_code(tmp_path, monkeypatch):
    config = ablation.make_config(8, 'uniform')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    split = tmp_path / config['args']['split_csv']
    split.parent.mkdir(parents=True)
    split.write_text('unused fixture')
    monkeypatch.setattr(ablation, 'ROOT', tmp_path)
    monkeypatch.setattr(ablation, 'source_fingerprint', lambda: config['source_sha256'])
    monkeypatch.setattr(ablation, 'digest', lambda path: config['data']['split_sha256'])

    class FailedProcess:
        stdout = ['CUDA fixture error\n']
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self):
            return 7

    monkeypatch.setattr(ablation.subprocess, 'Popen', lambda *args, **kwargs: FailedProcess())
    with pytest.raises(RuntimeError, match='Training failed'):
        ablation.run_config(path, 'python')
    log = (tmp_path / config['log']).read_text()
    assert '"num_frames": 8' in log
    assert '"seed": 42' in log
    assert 'CUDA fixture error' in log
    assert 'EXIT_CODE: 7' in log
