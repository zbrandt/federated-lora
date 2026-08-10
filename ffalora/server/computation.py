# Have each selected client train locally, and collect all of their results.
from __future__ import annotations

import torch
from transformers import PreTrainedModel

from ffalora.client import Client


def client_computation(
        selected_clients: list[Client],
        model: PreTrainedModel,
        global_state: dict[str, torch.Tensor],
    ) -> tuple[list[dict[str, torch.Tensor]], list[float], list[int], int, list[float | None]]:
    states = []
    losses = []
    weights = []
    uploaded = 0
    epsilons: list[float | None] = []

    for client in selected_clients:
        result = client.local_update(model, global_state)
        states.append(result.state_dict)
        losses.append(result.average_loss)
        weights.append(result.n_examples)
        uploaded += result.uploaded_bytes
        epsilons.append(result.epsilon_spent)

    return (states, losses, weights, uploaded, epsilons)
