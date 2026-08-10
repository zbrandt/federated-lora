from __future__ import annotations

import torch
from torch.nn import Module

from federated_lora.server import Server


class DPLoRA:
	name = 'dp_lora'

	def set_target_modules(self, model: Module) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_' in name or 'classifier' in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def local_update(
		self, 
		model: Module, 
		round: int
	) -> None:
		return None

	def aggregate(
		self, 
		uploads: list[dict[str, torch.Tensor]], 
		round: int
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads)
