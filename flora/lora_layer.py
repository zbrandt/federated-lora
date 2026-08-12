from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, lora_alpha: float, lora_dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.rank = rank
        self.scaling = lora_alpha / rank
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.A = nn.Linear(in_features, rank, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.B(self.A(self.dropout(x)))
        return base_out + lora_out * self.scaling


def inject_lora(model: nn.Module, config) -> None:
    target_names = set(config.target_modules)

    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name not in target_names or not isinstance(child, nn.Linear):
                continue

            layer = LoRALayer(
                base_layer=child,
                rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            layer = layer.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, child_name, layer)
