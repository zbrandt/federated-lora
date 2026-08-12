# Average client updates together, weighted by how many examples each client trained on.
from __future__ import annotations

import torch


def aggregate(state_dicts: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    # A is identical and frozen for every client, so this weighted average is exact --
    # no cross-term correction needed the way FLoRA's stacking approach required.
    # Note: clients with a lower rank contribute zeros in their unused B columns, which
    # dilutes those columns' average whenever a mix of ranks is selected in the same round.
    total_weight = float(sum(weights))
    keys = state_dicts[0].keys()
    aggregated: dict[str, torch.Tensor] = {}

    for key in keys:
        reference = state_dicts[0][key]
        accumulator = torch.zeros_like(reference, dtype=torch.float32)
        for client_update, weight in zip(state_dicts, weights):
            accumulator.add_(client_update[key].to(dtype=torch.float32), alpha=weight / total_weight)
        aggregated[key] = accumulator.to(dtype=reference.dtype, device=reference.device)

    return aggregated
