from __future__ import annotations

import torch
import torch.nn as nn

# TODO: from torch.optim import Optimizer for non-DP runs
from opacus.optimizers.optimizer import DPOptimizer

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
		self, model: nn.Module, optimizer: DPOptimizer, round: int
	) -> None:
		if optimizer.pre_step():
			for name, parameter in model.named_parameters():
				if parameter.grad is None:
					continue
				if (round % 2 == 0 and 'lora_B' in name) or (
					round % 2 == 1 and 'lora_A' in name
				):
					parameter.grad.zero_()

			optimizer.original_optimizer.step()
			optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads, num_examples)
