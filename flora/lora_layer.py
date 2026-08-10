"""
This file creates the FFA-LoRA (Frozen-A LoRA) layer, which is a LoRA adapter with a frozen, shared `A_max` and a rank-heterogeneous, DP-SGD-trainable `B`. 
The `A_max` is a fixed, orthogonal-init buffer that is never trained or noised, while `B` is the only trainable component. 
The layer supports rank heterogeneity, allowing clients to have different active ranks for their `B` matrices. 
The `inject_heterogeneous_lora` function walks through the model and replaces specified submodules with the FFA-LoRA layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The FFA-LoRA layer, which wraps a base nn.Linear layer and adds a low-rank adaptation with a frozen A matrix and a trainable B matrix.
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

        # Create the frozen A_max matrix with orthogonal initialization. This matrix is shared across all clients and is never trained or noised.
        a_max = torch.empty(r_max, in_features)
        generator = torch.Generator().manual_seed(seed)
        try:
            nn.init.orthogonal_(a_max, generator=generator)
        except TypeError:
            # For PyTorch versions < 1.9, the generator argument is not available, so we manually set the RNG state.
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            nn.init.orthogonal_(a_max)
            torch.set_rng_state(rng_state)
        self.register_buffer("A_max", a_max, persistent=False)

        self.B = nn.Linear(r_max, out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    # Forward pass through the FFA-LoRA layer. Computes the output of the base layer and adds the low-rank adaptation from the active rows of A_max and the trainable B matrix.
    # parameters: x: torch.Tensor - The input tensor to the layer.
    # returns: torch.Tensor - The output tensor after applying the base layer and the low-rank adaptation.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        rank = self.active_rank
        a_active = self.A_max[:rank, :]
        projected = F.linear(self.dropout(x), a_active)
        if rank < self.r_max:
            projected = F.pad(projected, (0, self.r_max - rank))
        lora_out = self.B(projected)

        return base_out + lora_out * self.scaling

# Walks through the model and replaces every submodule whose attribute name matches `config.target_modules` (e.g. "query", "value") with a `HeterogeneousDPLoRALayer` wrapping the original `nn.Linear`, in place. 
# This allows for the injection of FFA-LoRA adapters into the model for federated learning experiments.
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
