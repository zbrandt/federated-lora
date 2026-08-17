from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.privacy import privatize, zero_grad
from federated_lora.server import Server


class RoLoRA:
	name = 'rolora'

	def set_target_modules(self, model: nn.Module) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_' in name or 'classifier' in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def step(
		self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
	) -> None:
		params = privatize(model, optimizer)
		if params is None:
			return

		# freeze A and update B in an odd communication round
		# freeze B and update A in an even communication round
		for name, parameter in model.named_parameters():
			if parameter.grad is None:
				continue
			if (round % 2 == 0 and 'lora_B' in name) or (
				round % 2 == 1 and 'lora_A' in name
			):
				parameter.grad.zero_()

		optimizer.original_optimizer.step()

		zero_grad(params)
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads, num_examples)
