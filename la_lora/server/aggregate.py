from __future__ import annotations

import torch


def aggregate(
	state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
	"""
	Perform server aggregation of client updates.

	Server calculates an unweighted mean of LoRA adapter tensors key-by-key
	across client updates.

	Parameters
	----------
	state_dicts : list[dict[str, torch.Tensor]]
		list of dictionaries of updated adapter states from each client

	Returns
	------
	dict[str, torch.Tensor]
		unweighted mean of each adapter across clients key-by-key
	"""
	n = len(state_dicts)
	averaged = {}
	for key, reference in state_dicts[0].items():
		# create a tensor of torch. float32 zeroes with same size as key reference
		accumulator = torch.zeros_like(reference, dtype=torch.float32)

		for client_update in state_dicts:
			accumulator.add_(client_update[key].to(dtype=torch.float32))

		averaged[key] = (accumulator / n).to(
			dtype=reference.dtype, device=reference.device
		)

	return averaged
