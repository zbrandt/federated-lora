from __future__ import annotations

import torch
from transformers import PreTrainedModel

from la_lora.client import Client


def client_computation(
	selected_clients: list[Client],
	model: PreTrainedModel,
	global_state: dict[str, torch.Tensor],
) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
	"""
	Compute each client's local update to the global model.

	Iterate over selected clients and call their local update method. Record
	each client's update state and losses for later aggregation.

	Parameters
	----------
	selected_clients : list[Client]
	    A list of selected clients for one round of the federated learning
	    algorithm.
	model : PreTrainedModel
	    TODO: evaluate this
	global_state : dict[str, torch.Tensor]
	    A dictionary of the global model's initial parameters before updating.

	Returns
	-------
	tuple[list[dict[str, torch.Tensor]], list[float], int]
	    A tuple of states and losses from all the client updates.
	"""
	states = []
	losses = []

	for client in selected_clients:
		result = client.local_update(model, global_state)
		states.append(result.state_dict)
		losses.append(result.average_loss)

	return (states, losses)
