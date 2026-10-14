import numpy as np
import torch

from src.evaluation import accuracy, confusion_matrix, macro_f1_from_confusion
from src.models import build_r2plus1d18
from src.training import evaluate


def test_metrics_from_predictions():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    matrix = confusion_matrix(y_true, y_pred, 3)
    np.testing.assert_array_equal(matrix, np.array([[1, 1, 0], [0, 2, 0], [1, 0, 0]]))
    assert accuracy(y_true, y_pred) == 0.6
    assert 0.0 <= macro_f1_from_confusion(matrix) <= 1.0


def test_r2plus1d18_offline_forward():
    model = build_r2plus1d18(num_classes=4, pretrained=False, freeze_backbone=True)
    assert model.fc.out_features == 4
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("fc."))
    x = torch.randn(2, 3, 4, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_evaluate_tiny_loader():
    class TinyModel(torch.nn.Module):
        def forward(self, videos):
            return torch.tensor([[3.0, 0.0], [0.0, 4.0]], device=videos.device)

    loader = [(torch.randn(2, 3, 4, 16, 16), torch.tensor([0, 1]))]
    result, pred, target = evaluate(TinyModel(), loader, torch.device("cpu"))
    assert result.accuracy == 1.0
    assert pred.tolist() == [0, 1]
    assert target.tolist() == [0, 1]


def test_frozen_batchnorm_stats_stay_fixed_while_head_learns():
    from src.training import train_one_epoch

    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    before_mean = model[0].running_mean.clone()
    before_var = model[0].running_var.clone()
    before_head = model[1].weight.detach().clone()
    optimizer = torch.optim.SGD(model[1].parameters(), lr=0.1)
    train_one_epoch(model, [(torch.randn(8, 3) + 10, torch.zeros(8, dtype=torch.long))],
                    optimizer, torch.device('cpu'))
    assert torch.equal(model[0].running_mean, before_mean)
    assert torch.equal(model[0].running_var, before_var)
    assert not torch.equal(model[1].weight, before_head)


def test_unfrozen_batchnorm_updates_statistics():
    from src.training import train_one_epoch

    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, [(torch.ones(4, 3) * 5, torch.zeros(4, dtype=torch.long))],
                    optimizer, torch.device('cpu'))
    assert model[0].running_mean.sum() > 0
