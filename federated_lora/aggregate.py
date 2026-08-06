from __future__ import annotations

import torch


def uniform_mean(
	client_uploads: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
	"""
	FedAvg with equal weight per client, i.e. the per-key mean 1/|C| sum_k.

	Parameters
	----------
	client_uploads : list[dict[str, torch.Tensor]]
		One state-dict snapshot per client, restricted to the federated
		(trainable) keys: the LoRA A/B factors and the classification head.

	Returns
	-------
	dict[str, torch.Tensor]
		The new global state: the unweighted per-key mean over clients.
	"""
	return {
		key: torch.stack(
			[upload[key].float() for upload in client_uploads], dim=0
		).mean(dim=0)
		for key in client_uploads[0]
	}


def weighted_mean(
	client_uploads: list[dict[str, torch.Tensor]],
	weights: list[float],
) -> dict[str, torch.Tensor]:
	"""
	FedAvg weighted by client data size: sum_k rho_k * theta_k, sum_k rho_k = 1.

	Parameters
	----------
	client_uploads : list[dict[str, torch.Tensor]]
		One trainable-key state-dict snapshot per client.
	weights : list[float]
		Per-client weights (e.g. number of local training examples), aligned
		with ``client_uploads``. Normalized internally to sum to one.

	Returns
	-------
	dict[str, torch.Tensor]
		The new global state: the data-size-weighted per-key mean over clients.
	"""
	total = float(sum(weights))
	coeffs = [w / total for w in weights]
	return {
		key: sum(
			coeff * upload[key].float()
			for coeff, upload in zip(coeffs, client_uploads, strict=True)
		)
		for key in client_uploads[0]
	}
