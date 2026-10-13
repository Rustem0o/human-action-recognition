"""Classification metrics for the project baseline."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        matrix[target, pred] += 1
    return matrix


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    f1_scores = []
    for class_id in range(matrix.shape[0]):
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")
