from __future__ import annotations

from typing import Dict

from torch import nn
from transformers import AutoModel


class VehicleAttributeTransformer(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: Dict[str, int],
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                head: nn.Linear(hidden, num_classes[head])
                for head in num_classes.keys()
            }
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {head: classifier(pooled) for head, classifier in self.heads.items()}
