"""Video classification model factories used by the project baseline."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.video import (
    R2Plus1D_18_Weights,
    Swin3D_T_Weights,
    r2plus1d_18,
    swin3d_t,
)


def build_r2plus1d18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Return an R(2+1)D-18 model with a new classifier head.

    The baseline uses Kinetics-400 weights when ``pretrained`` is true, freezes the
    feature extractor by default, and trains only the final linear layer on the
    10-class UCF101 subset. Set ``pretrained=False`` for offline tests.
    """
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")

    return model


def build_video_swin_t(
    num_classes: int, pretrained: bool = True, freeze_backbone: bool = True
) -> nn.Module:
    """Return a torchvision Video Swin-T with a classifier for ``num_classes``.

    Kinetics-400 pretrained weights are downloaded by torchvision on first use.
    When freezing the backbone, only the final classifier remains trainable.
    """
    weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
    model = swin3d_t(weights=weights)

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("head.")

    return model
