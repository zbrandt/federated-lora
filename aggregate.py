from __future__ import annotations

from dataclasses import dataclass

import torch

# averaging rule shared by the current two methods
# outpus a new dict with same keys containing weighted-averaged tensors
def weighted_average_state_dicts(
	state_dicts: list[dict[str, torch.Tensor]],
	weights: list[float],
) -> dict[str, torch.Tensor]:
	"""Computes a weighted mean of model tensors key-by-key across clients."""

	# input validation
	if not state_dicts:
		raise ValueError("state_dicts cannot be empty")
	if len(state_dicts) != len(weights):
		raise ValueError("state_dicts and weights must have the same length")

	total_weight = float(sum(weights)) # denominator for the weighted average
	if total_weight <= 0:
		raise ValueError("weights must sum to a positive value")

	# iterate over each tensor
	averaged_state: dict[str, torch.Tensor] = {}
	for key, reference_tensor in state_dicts[0].items():
		# accumulate in float32 for numerical stability, then restore dtype.
		# accumulator tensor initialized to zeros
		accumulator = torch.zeros_like(reference_tensor, dtype=torch.float32) 

		# accumulate weighted tensors
		for state_dict, weight in zip(state_dicts, weights, strict=True):
			accumulator.add_(state_dict[key].to(dtype=torch.float32), alpha=float(weight))
		
		# normalize and restore original tensor format
		averaged_state[key] = (accumulator / total_weight).to(
			dtype=reference_tensor.dtype,
			device=reference_tensor.device,
		)
	return averaged_state


@dataclass(slots=True)
class FedITAvg:
	# FedIT averages the full LoRA adapter state from all clients.
	def aggregate(
		self,
		state_dicts: list[dict[str, torch.Tensor]],
		weights: list[float],
	) -> dict[str, torch.Tensor]:
		return weighted_average_state_dicts(state_dicts, weights)


@dataclass(slots=True)
class FFAAvg:
	# FFA-LoRA differs mathematically in how the adapters are trained, but the
	# server-side aggregation of the communicated state is still a weighted mean.
	def aggregate(
		self,
		state_dicts: list[dict[str, torch.Tensor]],
		weights: list[float],
	) -> dict[str, torch.Tensor]:
		return weighted_average_state_dicts(state_dicts, weights)
