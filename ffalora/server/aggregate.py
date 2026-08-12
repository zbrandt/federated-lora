# Average client updates together, weighted by how many examples each client trained on.
from __future__ import annotations

import torch


def aggregate(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[int],
    ranks: list[int],
    previous_state: dict[str, torch.Tensor],
    strategy: str = "naive",
) -> dict[str, torch.Tensor]:
    # A is identical and frozen for every client, so weighted-averaging B is exact --
    # no cross-term correction needed the way FLoRA's stacking approach required.
    if strategy == "naive":
        return _aggregate_naive(state_dicts, weights)
    if strategy == "rank_aware":
        return _aggregate_rank_aware(state_dicts, weights, ranks, previous_state)
    raise ValueError(f"unknown aggregation strategy: {strategy!r}")


# Plain weighted average. Clients with a lower rank contribute zeros in their unused
# B columns, which dilutes those columns whenever a round mixes clients of different ranks.
def _aggregate_naive(state_dicts: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
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


# Fixes the naive scheme's dilution: each B column is averaged only over the clients whose
# rank actually covers it, instead of over every selected client. A column no one selected
# this round covers keeps its previous value instead of being pulled toward zero.
def _aggregate_rank_aware(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[int],
    ranks: list[int],
    previous_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    total_weight = float(sum(weights))
    keys = state_dicts[0].keys()
    aggregated: dict[str, torch.Tensor] = {}

    for key in keys:
        reference = state_dicts[0][key]

        if not key.endswith(".B.weight"):
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for client_update, weight in zip(state_dicts, weights):
                accumulator.add_(client_update[key].to(dtype=torch.float32), alpha=weight / total_weight)
            aggregated[key] = accumulator.to(dtype=reference.dtype, device=reference.device)
            continue

        r_max = reference.shape[1]
        accumulator = torch.zeros_like(reference, dtype=torch.float32)
        column_weight = torch.zeros(r_max, dtype=torch.float32)
        for client_update, weight, rank in zip(state_dicts, weights, ranks):
            value = client_update[key][:, :rank].to(dtype=torch.float32)
            accumulator[:, :rank] += value * weight
            column_weight[:rank] += weight

        covered = column_weight > 0
        averaged = accumulator / column_weight.clamp_min(1.0)
        previous = previous_state[key].to(dtype=torch.float32)
        result = torch.where(covered.unsqueeze(0), averaged, previous)
        aggregated[key] = result.to(dtype=reference.dtype, device=reference.device)

    return aggregated
