from __future__ import annotations

import torch


def aggregate(state_dicts: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    """
    Perform weighted FedAvg over client updates.

    Under FFA-LoRA, `A` is identical and frozen across every client (see
    lora_layer.py), so `mean(B_k) @ A == mean(B_k @ A)` exactly -- there are
    no cross terms to correct for the way FLoRA's stacking aggregation
    handled, and no rank growth to reset each round. Plain weighted
    averaging is therefore exact, not an approximation.

    Rank-heterogeneous clients (see client.py) have already zeroed their
    inactive `B` columns (rank r_i..r_max) before upload, so those columns
    are averaged in as literal zeros here -- this dilutes higher-rank
    columns proportionally to how many selected clients this round have
    lower rank (fewer non-zero contributors averaged in with zero
    contributors). That's a known, accepted limitation of this simple
    scheme; a rank-aware normalized average (dividing each column by the
    weight of only the clients whose rank covers it) would remove the
    dilution but isn't implemented here.

    Parameters
    ----------
    state_dicts : list[dict[str, torch.Tensor]]
        Each selected client's updated `B` (per layer) and classifier head
        state.
    weights : list[int]
        Each client's aggregation weight (`n_examples`), same order as
        `state_dicts`.

    Returns
    -------
    dict[str, torch.Tensor]
        The weighted-average `B`/classifier head state, same shape as any
        individual client's contribution.
    """
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
