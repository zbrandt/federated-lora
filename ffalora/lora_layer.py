"""
A LoRA adapter where only one of its two small matrices (B) is ever trained --
the other (A) is fixed and shared by everyone. That keeps client updates easy
to average together and keeps DP noise from multiplying out of control.
Clients can also use a smaller "rank" than the maximum, so some clients can run lighter adapters than others.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# A single LoRA adapter with a frozen shared A, a trainable B, and a per-client active rank.
class HeterogeneousDPLoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, r_max: int, lora_alpha: float, lora_dropout: float, seed: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.r_max = r_max
        self.active_rank = r_max
        self.scaling = lora_alpha / r_max
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Orthogonal rows so any prefix of r_i < r_max rows is still a well-conditioned projection.
        a_max = torch.empty(r_max, in_features)
        generator = torch.Generator().manual_seed(seed)
        try:
            nn.init.orthogonal_(a_max, generator=generator)
        except TypeError:
            # Older torch without a generator kwarg -- seed the global RNG temporarily instead.
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            nn.init.orthogonal_(a_max)
            torch.set_rng_state(rng_state)
        self.register_buffer("A_max", a_max, persistent=False)

        self.B = nn.Linear(r_max, out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    # Run the base layer plus this adapter's contribution, using only the active rank's rows of A.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        rank = self.active_rank
        a_active = self.A_max[:rank, :]
        projected = F.linear(self.dropout(x), a_active)
        if rank < self.r_max:
            projected = F.pad(projected, (0, self.r_max - rank))
        lora_out = self.B(projected)

        return base_out + lora_out * self.scaling


# Replace the target linear layers in a model with LoRA adapters, in place.
def inject_heterogeneous_lora(model: nn.Module, config) -> None:
    target_names = set(config.target_modules)

    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name not in target_names or not isinstance(child, nn.Linear):
                continue

            layer = HeterogeneousDPLoRALayer(
                base_layer=child,
                r_max=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                seed=config.seed,
            )
            layer = layer.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, child_name, layer)
