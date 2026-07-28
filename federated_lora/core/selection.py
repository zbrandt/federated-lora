from __future__ import annotations

import torch

from la_lora.client import Client


def select_clients(
	clients: list[Client], sample_rate: float, generator: torch.Generator
) -> list[Client]:
	"""
	Select a set of clients to participate in a round of the federated learning
	algorithm.

	Parameters
	----------
	clients : list[Client]
		A list of all participating clients to select from.
	sample_rate : float
		The proportion of all clients to select.
	generator : torch.Generator
		Random number generator for selection.

	Returns
	-------
	list[Client]
		A list of randomly selected clients.
	"""
	k = max(1, round(sample_rate * len(clients)))
	picks = torch.randperm(len(clients), generator=generator)[:k]
	return [clients[i] for i in picks.tolist()]
