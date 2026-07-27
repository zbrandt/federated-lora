from __future__ import annotations

import torch
from transformers import PreTrainedModel

from flora.client import Client


def client_computation(
        selected_clients: list[Client],
        model: PreTrainedModel,
        global_state: dict[str, torch.Tensor],
    ) -> tuple[list[dict[str, torch.Tensor]], list[float], int, list[float | None]]:
    """
    Compute each client's local update to the global model.

    Iterate over selected clients and call their local update method. Record
    each client's update state, loss, and (if DP is enabled) accumulated
    epsilon spend for later aggregation/logging.

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
    tuple[list[dict[str, torch.Tensor]], list[float], int, list[float | None]]
        A tuple of states, losses, uploaded bytes, and per-client epsilon
        spend (`None` entries when DP is off) from all the client updates.
    """
    states = []
    losses = []
    uploaded = 0
    epsilons: list[float | None] = []

    for client in selected_clients:
        result = client.local_update(model, global_state)
        states.append(result.state_dict)
        losses.append(result.average_loss)
        uploaded += result.uploaded_bytes
        epsilons.append(result.epsilon_spent)

    return (states, losses, uploaded, epsilons)
