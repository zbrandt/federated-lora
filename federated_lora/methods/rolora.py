from __future__ import annotations

import torch
from torch.nn import Module

from federated_lora.server import Server


class RoLoRA:
	name = 'rolora'

	def prepare_model(
		self,
		model: Module,
	) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_' in name or 'classifier' in name:
				parameter.requires_grad = True

	def local_update(
		self, 
		model: Module, 
		round: int
	) -> None:
		for name, parameter in model.named_parameters():
			grad_sample = getattr(parameter, 'grad_sample', None)
			if grad_sample is None:
				continue
			if round % 2 == 0 and 'lora_B' in name:
				parameter.grad_sample = torch.zeros_like(grad_sample)
			elif round % 2 == 1 and 'lora_A' in name:
				parameter.grad_sample = torch.zeros_like(grad_sample)

	def aggregate(
		self, 
		uploads: list[dict[str, torch.Tensor]], 
		round: int
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads)
