from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class VehicleAttributeCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        num_blocks: int,
        num_classes: Dict[str, int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = [in_channels] + [base_filters * (2**i) for i in range(num_blocks)]
        features = []
        for idx in range(num_blocks):
            features.append(conv_block(channels[idx], channels[idx + 1]))
        self.feature_extractor = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = channels[-1]
        self.heads = nn.ModuleDict(
            {
                head: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, num_classes[head]),
                )
                for head in num_classes.keys()
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.feature_extractor(x)
        pooled = self.pool(feats).flatten(1)
        return {head: classifier(pooled) for head, classifier in self.heads.items()}
