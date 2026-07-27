from __future__ import annotations

import torch


def aggregate(state_dicts: list[dict[str, torch.Tensor]], scaling: float) -> dict[str, torch.Tensor]:
	"""
	Perform FLoRA-style server aggregation of client updates.

	Ideal aggregation wants mean(B_k A_k). Vanilla FedAvg over the LoRA
	factors instead computes mean(B_k) @ mean(A_k), which for two clients
	expands to:

		(1/2)(B1+B2) @ (1/2)(A1+A2) = 1/4 (B1A1 + B1A2 + B2A1 + B2A2)

	The cross terms B1A2 and B2A1 are pure noise -- they pair one client's
	down-projection with another client's up-projection. FLoRA avoids this by
	concatenating every client's A along the rank dimension and every
	client's B along the rank dimension: B_cat @ A_cat reproduces
	sum_k scaling * B_k @ A_k EXACTLY, with no cross terms, because matrix
	multiplication distributes over block-concatenation. Each client's B is
	scaled by `scaling` (the LoRA alpha/r factor, applied here rather than in
	the client's forward pass since state dicts carry raw, unscaled A/B
	tensors) before concatenation, so the reconstruction already carries the
	right coefficient for server/update.py to fold directly into the base
	weights.

	The classifier head (`modules_to_save`) has no low-rank structure to
	preserve, so it is aggregated as a plain unweighted mean, same as
	la_lora's aggregate().

	Parameters
	----------
	state_dicts : list[dict[str, torch.Tensor]]
		list of dictionaries of updated adapter states from each client
	scaling : float
		the LoRA scaling factor (lora_alpha / lora_rank) applied to each
		client's B before concatenation

	Returns
	-------
	dict[str, torch.Tensor]
		stacked (rank-grown) lora_A/lora_B tensors and averaged classifier
		head, key-by-key across clients
	"""
	keys = state_dicts[0].keys()
	aggregated: dict[str, torch.Tensor] = {}

	for key in keys:
		if "lora_B" in key:
			reference = state_dicts[0][key]
			scaled = [s[key].to(dtype=torch.float32) * scaling for s in state_dicts]
			aggregated[key] = torch.cat(scaled, dim=1).to(dtype=reference.dtype)
		elif "lora_A" in key:
			aggregated[key] = torch.cat([s[key] for s in state_dicts], dim=0)
		elif "modules_to_save" in key:
			reference = state_dicts[0][key]
			accumulator = torch.zeros_like(reference, dtype=torch.float32)
			for client_update in state_dicts:
				accumulator.add_(client_update[key].to(dtype=torch.float32))
			aggregated[key] = (accumulator / len(state_dicts)).to(dtype=reference.dtype, device=reference.device)
		else:
			raise KeyError(f"Unexpected adapter key '{key}' -- expected lora_A/lora_B/modules_to_save.")

	return aggregated
