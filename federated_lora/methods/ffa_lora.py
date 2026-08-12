from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.server import Server

from federated_lora.aggregate import uniform_mean


class FFALoRA:
	name = 'ffa_lora'

	def set_target_modules(self, model: nn.Module) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_B' in name or 'classifier' in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def step(
		self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
	) -> None:
		optimizer.step()
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads, num_examples)
