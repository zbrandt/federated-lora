from __future__ import annotations

import math

import torch
from transformers import PreTrainedModel


def update_model(
    model: PreTrainedModel,
    aggregated_state: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """
    Fold this round's exact stacked LoRA update into the frozen base weights,
    then reinitialize fresh rank-r0 adapters for the next round.

    aggregate() concatenates every client's A along the rank dimension and B
    along the rank dimension, so the `lora_A`/`lora_B` tensors in
    `aggregated_state` have GROWN rank (r0 * num_clients_selected). That
    growth can't be broadcast back into the model's rank-r0 adapters
    directly, so instead each LoRA layer's exact
    delta_w = B_cat @ A_cat (already scaled -- see aggregate.py) is added
    in place to that layer's frozen base weight, and a freshly reinitialized
    rank-r0 adapter (Kaiming-uniform A, zero B, matching PEFT's own default
    LoRA init) is hand back for the next round to train from a clean slate.
    This is FLoRA's periodic merge-and-reinit mechanism.

    Parameters
    ----------
    model : PreTrainedModel
        The shared PEFT model whose LoRA layers get walked and updated.
    aggregated_state : dict[str, torch.Tensor]
        This round's aggregated (stacked-rank) adapter state, and the
        averaged classifier head, from aggregate().
    generator : torch.Generator
        The server's RNG (CPU), reused here for reproducible reinitialization.

    Returns
    -------
    dict[str, torch.Tensor]
        The next round's global adapter state: fresh rank-r0 lora_A/lora_B
        tensors plus the averaged classifier head.
    """
    next_state: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue

        a_key = f"{name}.lora_A.default.weight"
        b_key = f"{name}.lora_B.default.weight"
        if a_key not in aggregated_state or b_key not in aggregated_state:
            continue

        base_weight = module.base_layer.weight
        a_cat = aggregated_state[a_key].to(device=base_weight.device)
        b_cat = aggregated_state[b_key].to(device=base_weight.device)
        delta_w = b_cat.float() @ a_cat.float()

        with torch.no_grad():
            base_weight.add_(delta_w.to(base_weight.dtype))

            a_param = module.lora_A["default"].weight
            b_param = module.lora_B["default"].weight
            _reinit_lora_a(a_param, generator)
            b_param.zero_()

        next_state[a_key] = a_param.detach().cpu().clone()
        next_state[b_key] = b_param.detach().cpu().clone()

    for key, value in aggregated_state.items():
        if "modules_to_save" in key:
            next_state[key] = value

    return next_state


def _reinit_lora_a(tensor: torch.Tensor, generator: torch.Generator) -> None:
    """
    Kaiming-uniform init matching PEFT's default `reset_lora_parameters`
    (`nn.init.kaiming_uniform_(weight, a=sqrt(5))`, which reduces to
    `bound = 1 / sqrt(fan_in)` for a 2D weight), but driven by our own CPU
    generator for reproducibility. `nn.init.kaiming_uniform_` itself doesn't
    accept a `generator` argument, and a CPU `torch.Generator` cannot drive
    `.uniform_` on a CUDA tensor directly -- so the draw happens on CPU and is
    then copied onto `tensor`'s actual device.
    """
    fan_in = tensor.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    cpu_values = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu").uniform_(-bound, bound, generator=generator)
    tensor.copy_(cpu_values.to(tensor.device))
