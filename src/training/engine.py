"""Small training and evaluation loops for video classifiers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    seconds: float


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_one_epoch(model, loader, optimizer, device, criterion=None) -> EpochResult:
    """Train for one epoch and return average loss/accuracy."""
    criterion = criterion or torch.nn.CrossEntropyLoss()
    model.train()
    # requires_grad=False does not freeze BatchNorm running statistics.
    # Preserve the pretrained backbone while its classifier is being trained.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.affine and not any(p.requires_grad for p in module.parameters()):
                module.eval()
    start = time.perf_counter()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += batch_size

    if total_seen == 0:
        raise ValueError("Training loader is empty")
    seconds = time.perf_counter() - start
    return EpochResult(total_loss / total_seen, total_correct / total_seen, seconds)


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None) -> tuple[EpochResult, torch.Tensor, torch.Tensor]:
    """Evaluate and return metrics plus concatenated predictions and labels."""
    criterion = criterion or torch.nn.CrossEntropyLoss()
    model.eval()
    start = time.perf_counter()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    predictions = []
    targets = []

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(videos)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((pred == labels).sum().item())
        total_seen += batch_size
        predictions.append(pred.cpu())
        targets.append(labels.cpu())

    if total_seen == 0:
        raise ValueError("Evaluation loader is empty")
    seconds = time.perf_counter() - start
    result = EpochResult(total_loss / total_seen, total_correct / total_seen, seconds)
    return result, torch.cat(predictions), torch.cat(targets)
