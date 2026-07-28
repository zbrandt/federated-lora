from __future__ import annotations

import torch
from transformers import PreTrainedModel

from federated_lora.core.client import Client
from federated_lora.methods.base import Method


def client_computation(
	method: Method,
	selected_clients: list[Client],
	model: PreTrainedModel,
	global_state: dict[str, torch.Tensor],
	round_index: int,
) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
	"""
	Compute each client's local update to the global model.

	Iterate over selected clients and call their local update method. Record
	each client's update state and losses for later aggregation.

	Parameters
	----------
	method : Method
		TODO
	selected_clients : list[Client]
		A list of selected clients for one round of the federated learning
		algorithm.
	model : PreTrainedModel
		The shared PEFT model passed to each client's ``local_update``. Clients
		load ``global_state`` into it.
	global_state : dict[str, torch.Tensor]
		A dictionary of the global model's initial parameters before updating.
	round_index : int
		The global round index used for seeding client's data shuffles.

	Returns
	-------
	tuple[list[dict[str, torch.Tensor]], list[float], int]
		A tuple of states and losses from all the client updates.
	"""
	states = []
	losses = []

	for client in selected_clients:
		result = method.local_update(client, model, global_state, round_index)
		states.append(result.state_dict)
		losses.append(result.average_loss)

	return (states, losses)
