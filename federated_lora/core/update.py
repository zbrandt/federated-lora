from __future__ import annotations

import torch


def update_model(
	aggregated_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
	"""
	Update the global model as part of the federated learning algorithm.

	For LoRA, this simply returns the aggregation of the local client updates.

	Parameters
	----------
	aggregated_state : dict[str, torch.Tensor]
		Aggregation result from client updates.

	Returns
	-------
	dict[str, torch.Tensor]
		A dictionary of each parameter and its corresponding aggregated tensor.
	"""
	return aggregated_state
