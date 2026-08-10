from __future__ import annotations

import torch
from transformers import PreTrainedModel

from ffalora.client import Client


def client_computation(
        selected_clients: list[Client],
        model: PreTrainedModel,
        global_state: dict[str, torch.Tensor],
    ) -> tuple[list[dict[str, torch.Tensor]], list[float], list[int], int, list[float | None]]:
    """
    Compute each client's local update to the global model.

    Iterate over selected clients and call their local update method. Record
    each client's update state, loss, aggregation weight (`n_examples`),
    uploaded bytes, and (if that client has DP enabled) accumulated epsilon
    spend for later aggregation/logging.

    Parameters
    ----------
    selected_clients : list[Client]
        A list of selected clients for one round of the federated learning
        algorithm.
    model : PreTrainedModel
        The shared model instance every client trains in place, one at a time.
    global_state : dict[str, torch.Tensor]
        A dictionary of the global model's initial parameters before updating.

    Returns
    -------
    tuple[list[dict[str, torch.Tensor]], list[float], list[int], int, list[float | None]]
        A tuple of states, losses, aggregation weights, uploaded bytes, and
        per-client epsilon spend (`None` entries when that client's DP is
        off) from all the client updates.
    """
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
