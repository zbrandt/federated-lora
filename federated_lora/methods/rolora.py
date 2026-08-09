from __future__ import annotations

import torch
from torch.nn import Module


class RoLoRA:
	name = 'rolora'

	def local_update(self, model: Module, round: int) -> None:
		for name, parameter in model.named_parameters():
			grad_sample = getattr(parameter, 'grad_sample', None)
			if grad_sample is None:
				continue
			if round % 2 == 0 and 'lora_B' in name:
				parameter.grad_sample = torch.zeros_like(grad_sample)
			elif round % 2 == 1 and 'lora_A' in name:
				parameter.grad_sample = torch.zeros_like(grad_sample)


	def aggregate(
		self, uploads: list[dict[str, torch.Tensor]], round: int
	) -> dict[str, torch.Tensor]:
		agg = {}

		for upload in uploads:
			for key, value in upload.items():
				# TODO: determine if skipping A vs B matrix in aggregation based
				# on round index is critical or not

				if key not in agg:
					agg[key] = value.clone().float()
				else:
					agg[key] += value.float()

		for key in agg:
			agg[key] /= float(len(uploads))

		return agg
